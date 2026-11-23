"""Thresholds builder and Gate 1 runner on small seeded synthetic stores. No network, no real data."""

from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import station_ids, write_config

from wrfcomp.harness.config import load_config
from wrfcomp.harness.schemas import Scorecard, validate_thresholds

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
VARIABLES = {  # name -> (mean, seasonal amplitude, diurnal amplitude, noise sd)
    "t2m": (285.0, 10.0, 4.0, 2.0),
    "td2m": (279.0, 8.0, 1.5, 2.0),
    "u10": (1.0, 0.5, 0.5, 2.5),
    "v10": (-0.5, 0.5, 0.5, 2.5),
    "msl": (101500.0, 300.0, 50.0, 500.0),
}
UNITS = {"t2m": "K", "td2m": "K", "u10": "m/s", "v10": "m/s", "msl": "Pa"}
N_STATIONS = 6
PERIODS = ("2024-01", "2024-07")
GEFS_LEADS = (3, 6, 12, 24, 48)
MODELS = ("climatology", "persistence", "gefs")
# One more than the gate's member floor, so the synthetic GEFS is always eligible for the gate.
GEFS_MEMBERS = load_config().gate.min_members_for_gate + 1


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


build_thresholds = load_script("build_thresholds")
run_gate1 = load_script("run_gate1")


# --------------------------------------------------------------------------- synthetic world


def _station_coords() -> dict:
    return {
        "latitude": ("station_id", np.linspace(30.0, 45.0, N_STATIONS)),
        "longitude": ("station_id", np.linspace(-120.0, -75.0, N_STATIONS)),
        "elevation_m": ("station_id", np.linspace(5.0, 1500.0, N_STATIONS)),
    }


def make_truth(start: str, end: str, seed: int) -> tuple[xr.Dataset, dict[str, np.ndarray]]:
    """Hourly truth in the contract shape with the `X_offset_min` companions (addendum 1).

    Season + time of day + slowly varying weather. About 3 % of the values at hours 4, 12 and 20 UTC
    are missing, never two hours in a row, so persistence always finds a value within its 3 hours.
    Also returns the values before the gaps were cut, for building a synthetic "GEFS".
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="1h")
    n = len(times)
    season = np.cos(2 * np.pi * (times.day_of_year.values - 200) / 365.25)[:, None]
    daily = np.cos(2 * np.pi * (times.hour.values - 15) / 24)[:, None]
    gap = (times.hour.values % 8 == 4)[:, None] & (rng.random((n, N_STATIONS)) < 0.03)
    data, clean = {}, {}
    for name, (mean, seasonal, diurnal, sd) in VARIABLES.items():
        weather = np.zeros((n, N_STATIONS))
        shocks = rng.normal(0, sd * np.sqrt(1 - 0.97**2), (n, N_STATIONS))
        weather[0] = rng.normal(0, sd, N_STATIONS)
        for i in range(1, n):
            weather[i] = 0.97 * weather[i - 1] + shocks[i]
        full = mean + seasonal * season + diurnal * daily + weather + np.linspace(-1, 1, N_STATIONS)
        clean[name] = full
        values = np.where(gap, np.nan, full).astype("float32")
        minutes = np.where(rng.random((n, N_STATIONS)) < 0.8, -7, 4)
        data[name] = (("valid_time", "station_id"), values, {"units": UNITS[name]})
        data[f"{name}_offset_min"] = (
            ("valid_time", "station_id"),
            np.where(gap, -32768, minutes).astype("int16"),
        )
    coords = {
        "valid_time": times.values.astype("datetime64[ns]"),
        "station_id": station_ids(N_STATIONS),
        **_station_coords(),
    }
    return xr.Dataset(data, coords=coords), clean


def write_gefs_cycles(root: Path, truth: xr.Dataset, clean: dict[str, np.ndarray], seed: int) -> None:
    """One store per cycle, laid out like the real extractor's output (Zarr v2, one init time)."""
    rng = np.random.default_rng(seed)
    times = truth["valid_time"].values
    leads = np.array(GEFS_LEADS, dtype="timedelta64[h]").astype("timedelta64[ns]")
    for label in PERIODS:
        start = np.datetime64(f"{label}-01T00", "ns")
        end = start + np.timedelta64(31, "D") - np.timedelta64(1, "h")
        for init in run_gate1.evaluation_init_times(start, end, [0, 12]):
            rows = np.searchsorted(times, init + leads)
            data = {}
            for name, (_, _, _, sd) in VARIABLES.items():
                error_sd = 0.25 * sd * np.sqrt(np.array(GEFS_LEADS) / 12.0)[:, None]
                centre = clean[name][rows] + rng.normal(0, 1, (len(leads), N_STATIONS)) * error_sd
                noise = rng.normal(0, 1, (len(leads), N_STATIONS, GEFS_MEMBERS))
                members = centre[..., None] + noise * error_sd[..., None]
                dims = ("init_time", "lead_time", "station_id", "ens_member")
                data[name] = (dims, members[None].astype("float32"), {"units": UNITS[name]})
            coords = {
                "init_time": np.array([init], dtype="datetime64[ns]"),
                "lead_time": leads,
                "station_id": station_ids(N_STATIONS).astype(str),
                "ens_member": np.arange(GEFS_MEMBERS),
                **_station_coords(),
            }
            ds = xr.Dataset(data, coords=coords, attrs={"model_id": "noaa-gefs", "model_version": "1"})
            store = root / f"{pd.Timestamp(init):%Y%m%d%H}.zarr"
            ds.to_zarr(store, mode="w", consolidated=True, zarr_format=2)


def build_world(root: Path) -> dict[str, Path]:
    data = root / "data"
    (data / "truth").mkdir(parents=True)
    (data / "baselines" / "gefs").mkdir(parents=True)
    train, _ = make_truth("2022-12-15T00", "2023-08-15T00", seed=1)
    # The scored year's truth starts one day early. Persistence at the first init time (1 January 00Z)
    # needs an observation taken at or before that time; a file starting exactly at 00Z only has it
    # for stations whose 00Z report was not taken after the hour.
    truth, clean = make_truth("2023-12-31T00", "2024-08-03T00", seed=2)
    train.to_netcdf(data / "truth" / "isd_conus_train.nc", engine="h5netcdf")
    truth.to_netcdf(data / "truth" / "isd_conus_2024.nc", engine="h5netcdf")
    write_gefs_cycles(data / "baselines" / "gefs", truth, clean, seed=3)
    config = write_config(root, list(station_ids(N_STATIONS)))
    return {"root": root, "data": data, "config": config}


@pytest.fixture(scope="module")
def base_world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Inputs plus the thresholds file. Never modified: tests that change things work on a copy."""
    world = build_world(tmp_path_factory.mktemp("gate1"))
    code = build_thresholds.main(
        [
            "--train-truth", str(world["data"] / "truth" / "isd_conus_train.nc"),
            "--out", str(world["data"] / "thresholds" / "tail_p95.nc"),
            "--config", str(world["config"]),
        ]
    )  # fmt: skip
    assert code == 0
    return world


def copy_world(base: dict[str, Path], dest: Path) -> dict[str, Path]:
    shutil.copytree(base["root"], dest)
    return {"root": dest, "data": dest / "data", "config": dest / "configs" / "splits.yaml"}


def run(world: dict[str, Path], out: Path, *extra: str, periods: tuple[str, ...] = PERIODS) -> int:
    args = ["--data-dir", str(world["data"]), "--init-hours", "0", "12", "--out", str(out)]
    for label in periods:
        args += ["--period", label]
    args += ["--config", str(world["config"]), "--allow-tmp-data-dir", *extra]
    return run_gate1.main(args)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the Gate 1 scripts must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


# --------------------------------------------------------------------------- build_thresholds


def test_thresholds_file_passes_the_schema_validator(base_world: dict[str, Path], no_network: None) -> None:
    cfg = load_config()
    path = base_world["data"] / "thresholds" / "tail_p95.nc"
    with xr.open_dataset(path) as thresholds:
        thresholds.load()
    validate_thresholds(thresholds, variables=cfg.evaluation.variables.primary)
    assert set(thresholds.data_vars) == set(cfg.evaluation.variables.primary)  # no offset companions
    assert thresholds.sizes == {"station_id": N_STATIONS, "month": 12}
    assert thresholds["month"].dtype == np.int64
    assert thresholds.attrs["tail_quantile"] == cfg.gate.tail_quantile == 0.95
    assert all(thresholds[name].dtype == np.float32 for name in thresholds.data_vars)

    with xr.open_dataset(base_world["data"] / "truth" / "isd_conus_train.nc") as train:
        january = train["t2m"].sel(valid_time=train["valid_time"].dt.month == 1).values.astype(np.float64)
    expected = np.nanquantile(january, 0.95, axis=0).astype(np.float32)
    np.testing.assert_array_equal(thresholds["t2m"].sel(month=1).values, expected)
    # the training sample has no September to November: no threshold there, and it says so with NaN
    assert np.isnan(thresholds["t2m"].sel(month=[9, 10, 11]).values).all()
    assert np.isfinite(thresholds["t2m"].sel(month=[1, 2, 7, 8, 12]).values).all()


def test_thresholds_refuse_input_past_the_train_split(base_world: dict[str, Path], tmp_path: Path) -> None:
    truth_dir = base_world["data"] / "truth"
    config = ["--config", str(base_world["config"])]
    out = tmp_path / "thresholds.nc"
    # the scored year's truth is validation data
    assert (
        build_thresholds.main(
            ["--train-truth", str(truth_dir / "isd_conus_2024.nc"), "--out", str(out), *config]
        )
        == 2
    )
    assert not out.exists()
    # one hour past the train split end is enough; exactly the end is fine
    for end, expected in (("2023-12-17T18:00", 0), ("2023-12-17T19:00", 2)):
        late, _ = make_truth("2023-12-10T00", end, seed=5)
        late.to_netcdf(tmp_path / f"late{expected}.nc", engine="h5netcdf")
        target = tmp_path / f"out{expected}.nc"
        code = build_thresholds.main(
            ["--train-truth", str(tmp_path / f"late{expected}.nc"), "--out", str(target), *config]
        )
        assert code == expected
        assert target.exists() == (expected == 0)
    # mixed: a clean store next to a late one is still refused
    both = [str(truth_dir / "isd_conus_train.nc"), str(tmp_path / "late2.nc")]
    assert build_thresholds.main(["--train-truth", *both, "--out", str(out), *config]) == 2
    assert not out.exists()
    assert not list(tmp_path.glob(".*partial*"))


def test_thresholds_split_input_and_existing_output(base_world: dict[str, Path], tmp_path: Path) -> None:
    config = ["--config", str(base_world["config"])]
    whole = base_world["data"] / "truth" / "isd_conus_train.nc"
    with xr.open_dataset(whole) as train:
        train.isel(valid_time=slice(0, 3000)).to_netcdf(tmp_path / "a.nc", engine="h5netcdf")
        train.isel(valid_time=slice(3000, None)).to_zarr(tmp_path / "b.zarr", consolidated=False)
    out = tmp_path / "split.nc"
    parts = [str(tmp_path / "b.zarr"), str(tmp_path / "a.nc")]  # order must not matter
    assert build_thresholds.main(["--train-truth", *parts, "--out", str(out), *config]) == 0
    with (
        xr.open_dataset(out) as got,
        xr.open_dataset(base_world["data"] / "thresholds" / "tail_p95.nc") as want,
    ):
        xr.testing.assert_identical(got.load(), want.load())
    # an existing file is not replaced silently
    before = out.read_bytes()
    assert build_thresholds.main(["--train-truth", str(whole), "--out", str(out), *config]) == 2
    assert out.read_bytes() == before
    assert (
        build_thresholds.main(["--train-truth", str(whole), "--out", str(out), "--overwrite", *config]) == 0
    )
    overlap = [str(whole), str(tmp_path / "a.nc")]
    assert build_thresholds.main(["--train-truth", *overlap, "--out", str(tmp_path / "x.nc"), *config]) == 2
    assert (
        build_thresholds.main(
            ["--train-truth", str(tmp_path / "nope.nc"), "--out", str(tmp_path / "y.nc"), *config]
        )
        == 2
    )


# --------------------------------------------------------------------------- run_gate1


def test_runner_writes_three_scorecards_per_period_and_rerun_is_identical(
    base_world: dict[str, Path], tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture
) -> None:
    world = copy_world(base_world, tmp_path / "world")
    out = tmp_path / "experiments"
    assert run(world, out) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(f"{model}_{period}.json" for model in MODELS for period in PERIODS)
    first = {name: (out / name).read_bytes() for name in names}

    for period in PERIODS:
        cards = {m: Scorecard.model_validate_json(first[f"{m}_{period}.json"]) for m in MODELS}
        month = int(period[-2:])
        for model, card in cards.items():
            assert card.n_stations == N_STATIONS
            assert card.n_init_times == 62 and card.init_hours_scored == [0, 12]
            assert (card.period_start.month, card.period_start.day, card.period_start.hour) == (month, 1, 0)
            assert (card.period_end.month, card.period_end.day, card.period_end.hour) == (month, 31, 12)
            assert card.sealed_data_used is False
            assert set(card.metrics) == set(VARIABLES)
            assert all(n > 0 for n in card.n_pairs.values()), model
        assert cards["climatology"].model_id == "climatology"
        assert cards["climatology"].baseline_id is None and cards["climatology"].passed_gate is False
        assert cards["climatology"].leads_scored == [1, 3, 6, 12, 24, 48]
        assert cards["persistence"].baseline_id == "climatology"
        assert cards["persistence"].leads_scored == [1, 3, 6, 12, 24, 48]
        assert cards["gefs"].model_id == "noaa-gefs" and cards["gefs"].baseline_id == "climatology"
        assert cards["gefs"].leads_scored == list(GEFS_LEADS)  # the leads the model has
        # all three were scored against the same truth and thresholds, the last two against one baseline
        assert len({c.truth_sha256 for c in cards.values()}) == 1
        assert len({c.thresholds_sha256 for c in cards.values()}) == 1
        assert cards["persistence"].baseline_sha256 == cards["gefs"].baseline_sha256
        assert cards["persistence"].baseline_sha256 == cards["climatology"].forecast_sha256
        # the synthetic "GEFS" knows the weather: it must beat climatology clearly
        assert cards["gefs"].metrics["t2m"]["all"].crpss_vs_baseline > 0.3
        assert cards["persistence"].metrics["t2m"]["1"].crpss_vs_baseline > 0.3

    # rerun in place: forecast stores are reused, every scorecard has the same bytes
    capsys.readouterr()
    assert run(world, out) == 0
    text = capsys.readouterr().out
    assert text.count("forecast store present, reused") == 6 and text.count("unchanged") == 6
    assert {name: (out / name).read_bytes() for name in names} == first

    # rebuild everything from the inputs into a new place: still the same bytes
    shutil.rmtree(world["data"] / "gate1")
    again = tmp_path / "experiments-again"
    assert run(world, again) == 0
    assert {name: (again / name).read_bytes() for name in names} == first
    assert not list(out.glob(".*")) and not list(again.glob(".*"))


def test_runner_resumes_and_guards_existing_scorecards(
    base_world: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    world = copy_world(base_world, tmp_path / "world")
    out = tmp_path / "experiments"
    gefs_dir = world["data"] / "baselines" / "gefs"
    held_back = tmp_path / "2024011512.zarr"
    shutil.move(gefs_dir / "2024011512.zarr", held_back)

    # one GEFS cycle is not downloaded yet: the other two models are still scored
    capsys.readouterr()
    assert run(world, out, periods=("2024-01",)) == 1
    text = capsys.readouterr().out
    assert "1 of 62 GEFS cycle store(s) missing" in text and "2024011512.zarr" in text
    assert sorted(p.name for p in out.iterdir()) == ["climatology_2024-01.json", "persistence_2024-01.json"]
    assert not (world["data"] / "gate1" / "2024-01_h00-12" / "gefs.zarr").exists()

    # the cycle arrives: the rerun only adds what was missing
    shutil.move(held_back, gefs_dir / "2024011512.zarr")
    kept = (out / "climatology_2024-01.json").read_bytes()
    assert run(world, out, periods=("2024-01",)) == 0
    assert (out / "gefs_2024-01.json").is_file()
    assert (out / "climatology_2024-01.json").read_bytes() == kept

    # a scorecard that no longer matches is reported, not overwritten
    card = json.loads(kept)
    card["metrics"]["t2m"]["all"]["crps"] = 0.0
    tampered = json.dumps(card).encode()
    (out / "climatology_2024-01.json").write_bytes(tampered)
    assert run(world, out, "--models", "climatology", periods=("2024-01",)) == 1
    assert (out / "climatology_2024-01.json").read_bytes() == tampered
    assert run(world, out, "--models", "climatology", "--overwrite", periods=("2024-01",)) == 0
    assert (out / "climatology_2024-01.json").read_bytes() == kept

    # a forecast store left over from another request is refused, not silently reused
    shutil.rmtree(world["data"] / "gate1" / "2024-01_h00-12" / "gefs.zarr")
    shutil.copytree(
        world["data"] / "gate1" / "2024-01_h00-12" / "persistence.zarr",
        world["data"] / "gate1" / "2024-01_h00-12" / "gefs.zarr",
    )
    capsys.readouterr()
    assert run(world, out, "--models", "gefs", periods=("2024-01",)) == 1
    assert "does not match this request" in capsys.readouterr().out


def test_gaps_are_masked_alike_and_a_nan_forecast_fails_only_its_model(
    base_world: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    world = copy_world(base_world, tmp_path / "world")
    # 1. Station 0 reports nothing from 09Z to 12Z on 10 January, so persistence has no observation
    #    within 3 hours of that 12Z run. The baselines fill such a gap themselves (version 3).
    path = world["data"] / "truth" / "isd_conus_2024.nc"
    with xr.open_dataset(path) as ds:
        truth = ds.load()
    outage = slice("2024-01-10T09", "2024-01-10T12")
    stations = truth["station_id"].values
    for name in VARIABLES:
        truth[name].loc[{"valid_time": outage, "station_id": stations[0]}] = np.nan
        truth[f"{name}_offset_min"].loc[{"valid_time": outage, "station_id": stations[0]}] = -32768
    path.unlink()
    truth.to_netcdf(path, engine="h5netcdf")
    # 2. Station 1 has no t2m threshold for January (no training data that month): not scored there.
    path = world["data"] / "thresholds" / "tail_p95.nc"
    with xr.open_dataset(path) as ds:
        thresholds = ds.load()
    thresholds["t2m"].loc[{"station_id": stations[1], "month": 1}] = np.nan
    path.unlink()
    thresholds.to_netcdf(path, engine="h5netcdf")
    # 3. One GEFS cycle has a NaN member at a point that is scored.
    store = world["data"] / "baselines" / "gefs" / "2024012000.zarr"
    with xr.open_zarr(store, chunks=None, consolidated=True) as ds:
        cycle = ds.load()
    cycle["u10"].values[0, 2, 3, 0] = np.nan
    shutil.rmtree(store)
    for name in cycle.variables:
        cycle[name].encoding.clear()
    cycle.to_zarr(store, mode="w", consolidated=True, zarr_format=2)

    out = tmp_path / "experiments"
    capsys.readouterr()
    assert run(world, out, periods=("2024-01",)) == 1
    captured = capsys.readouterr()
    # the NaN rule fails GEFS (exit code 3) and only GEFS
    assert "gefs         FAILED    wrfcomp-eval exit code 3" in captured.out
    assert "candidate forecast is not finite at 1 point(s)" in captured.err
    assert sorted(p.name for p in out.iterdir()) == ["climatology_2024-01.json", "persistence_2024-01.json"]

    cards = {m: json.loads((out / f"{m}_2024-01.json").read_text()) for m in ("climatology", "persistence")}
    for variable in VARIABLES:
        # one mask for every model: the same points are scored and the same points are left out
        assert cards["persistence"]["n_pairs"][variable] == cards["climatology"]["n_pairs"][variable] > 0
        masked = {m: c["metrics"][variable]["all"]["n_masked_no_threshold"] for m, c in cards.items()}
        assert masked["persistence"] == masked["climatology"]
        assert (masked["climatology"] > 0) == (variable == "t2m")
    assert cards["climatology"]["n_pairs"]["t2m"] < cards["climatology"]["n_pairs"]["u10"]


def test_runner_refuses_bad_requests_before_reading_data(
    base_world: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "experiments"
    for periods in (
        ("2024-12",),
        ("2023-07",),
        ("2025-01",),
        ("2024-1",),
        ("2024-13",),
        ("2024-01", "2024-01"),
    ):
        assert run(base_world, out, periods=periods) == 2, periods
    capsys.readouterr()
    assert run(base_world, out, periods=("2024-12",)) == 2
    assert "validation split" in capsys.readouterr().err

    args = ["--data-dir", str(base_world["data"]), "--period", "2024-01", "--out", str(out)]
    args += ["--config", str(base_world["config"])]
    assert run_gate1.main([*args, "--init-hours", "0", "12"]) == 2  # a data dir under /tmp
    assert run_gate1.main([*args, "--init-hours", "3", "--allow-tmp-data-dir"]) == 2  # not an init hour
    repo = Path(__file__).resolve().parents[1]
    inside = [
        "--data-dir",
        str(repo / "experiments"),
        "--period",
        "2024-01",
        "--init-hours",
        "0",
        "--out",
        str(out),
    ]
    assert run_gate1.main(inside) == 2
    assert not out.exists() or not list(out.iterdir())
    assert not (base_world["data"] / "gate1").exists()


def test_runner_refuses_sealed_and_short_truth(base_world: dict[str, Path], tmp_path: Path, capsys) -> None:
    out = tmp_path / "experiments"
    # truth that reaches into the sealed split is refused on its coordinates
    sealed, _ = make_truth("2024-12-30T00", "2025-01-01T00", seed=7)
    sealed.to_netcdf(tmp_path / "sealed.nc", engine="h5netcdf")
    capsys.readouterr()
    assert run(base_world, out, "--truth", str(tmp_path / "sealed.nc"), periods=("2024-01",)) == 2
    assert "past the allowed end" in capsys.readouterr().err
    # truth that stops before the period's last valid time would silently score fewer points
    short, _ = make_truth("2024-01-01T00", "2024-01-20T00", seed=8)
    short.to_netcdf(tmp_path / "short.nc", engine="h5netcdf")
    assert run(base_world, out, "--truth", str(tmp_path / "short.nc"), periods=("2024-01",)) == 2
    assert "needs" in capsys.readouterr().err
    assert run(base_world, out, "--thresholds", str(tmp_path / "missing.nc"), periods=("2024-01",)) == 2
    assert not (base_world["data"] / "gate1").exists()
