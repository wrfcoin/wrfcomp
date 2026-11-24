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


def write_gefs_cycles(
    root: Path,
    truth: xr.Dataset,
    clean: dict[str, np.ndarray],
    seed: int,
    periods: tuple[str, ...] = PERIODS,
) -> None:
    """One store per cycle, laid out like the real extractor's output (Zarr v2, one init time)."""
    rng = np.random.default_rng(seed)
    times = truth["valid_time"].values
    leads = np.array(GEFS_LEADS, dtype="timedelta64[h]").astype("timedelta64[ns]")
    for label in periods:
        start = np.datetime64(f"{label}-01T00", "ns")
        end = start + np.timedelta64(pd.Period(label).days_in_month, "D") - np.timedelta64(1, "h")
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
        # Every model's ensemble, as written to disk, scored and serialised, meets the configured
        # member floor, and the scorecard's min_members gate says so.
        floor = load_config().gate.min_members_for_gate
        for model, card in cards.items():
            for variable in VARIABLES:
                assert card.metrics[variable]["all"].n_members >= floor, (model, variable)
                assert card.gates[variable]["min_members"] is True, (model, variable)
        assert all(cards["gefs"].metrics[v]["all"].n_members == GEFS_MEMBERS for v in VARIABLES)
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
        ("2023-07",),
        ("2025-01",),
        ("2024-1",),
        ("2024-13",),
        ("2024-01", "2024-01"),
    ):
        assert run(base_world, out, periods=periods) == 2, periods
    capsys.readouterr()
    assert run(base_world, out, periods=("2023-12",)) == 2
    assert "before the validation split" in capsys.readouterr().err
    assert run(base_world, out, periods=("2025-01",)) == 2
    assert "no init time is inside the validation split" in capsys.readouterr().err

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


def _config_with(world: dict[str, Path], **replace: str) -> None:
    """Rewrite split boundaries in the world's own copy of splits.yaml (never the repository's)."""
    text = world["config"].read_text()
    for old, new in replace.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    world["config"].write_text(text)


def test_last_init_rule_with_the_real_config() -> None:
    cfg = load_config()
    longest = np.timedelta64(max(cfg.evaluation.lead_hours), "h")
    for month in range(1, 12):  # January to November: nothing is left out
        period = run_gate1.parse_period(f"2024-{month:02d}", [0, 12], cfg)
        days = pd.Period(f"2024-{month:02d}").days_in_month
        assert (period.init_times.size, period.n_dropped) == (2 * days, 0)
        assert period.init_times.max() == np.datetime64(f"2024-{month:02d}-{days:02d}T12", "ns")
    # December stops at the last init inside the validation split, for every model.
    december = run_gate1.parse_period("2024-12", [0, 12], cfg)
    assert december.init_times.min() == np.datetime64("2024-12-01T00", "ns")
    assert december.init_times.max() == np.datetime64("2024-12-17T12", "ns")
    assert (december.init_times.size, december.n_dropped) == (34, 28)
    assert december.cycles[-1] == "2024121712" and not any(c.startswith("2025") for c in december.cycles)
    assert (december.init_times <= cfg.splits.validation.end).all()
    assert (december.init_times + longest < cfg.splits.sealed_test.start).all()
    # The validation end is inclusive: an 18Z run on the last day is scored.
    assert run_gate1.parse_period("2024-12", [18], cfg).init_times.max() == cfg.splits.validation.end
    assert run_gate1.parse_period("2024-12", [0, 6, 12, 18], cfg).init_times.size == 17 * 4


def test_last_init_rule_sealed_lead_limit(tmp_path: Path) -> None:
    # Without an embargo the sealed-lead limit is the one that cuts: 2024-12-29T12Z + 48 h is the last
    # valid time before 2025, so 12-30 and 12-31 are left out.
    world = {"config": write_config(tmp_path, list(station_ids(N_STATIONS)))}
    _config_with(world, **{'end_utc: "2024-12-17T18:00:00Z"': 'end_utc: "2024-12-31T18:00:00Z"'})
    cfg = load_config(world["config"])
    period = run_gate1.parse_period("2024-12", [0, 12], cfg)
    assert period.init_times.max() == np.datetime64("2024-12-29T12", "ns")
    assert (period.init_times.size, period.n_dropped) == (58, 4)
    last_valid = period.init_times.max() + np.timedelta64(48, "h")
    assert last_valid == np.datetime64("2024-12-31T12", "ns") < cfg.splits.sealed_test.start
    # 12-30T00Z would verify at 2025-01-01T00Z, the first sealed hour: left out, not scored at shorter leads.
    assert np.datetime64("2024-12-30T00", "ns") not in period.init_times
    # The rule follows the longest configured lead, whatever leads a model has.
    kept = run_gate1.scorable_init_times(
        np.array(["2024-12-29T12", "2024-12-30T00", "2024-12-31T12"], dtype="datetime64[ns]"), cfg
    )
    assert kept.tolist() == np.array(["2024-12-29T12"], dtype="datetime64[ns]").tolist()


def test_runner_scores_a_period_that_stops_early_alike_for_all_models(
    base_world: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture, no_network: None
) -> None:
    world = copy_world(base_world, tmp_path / "w")
    # In this world the validation split ends on 17 July 18Z, as the real one does in December.
    _config_with(world, **{'end_utc: "2024-12-17T18:00:00Z"': 'end_utc: "2024-07-17T18:00:00Z"'})
    # The GEFS cycles after the last scored init are removed: the runner must not need or open them.
    removed = 0
    for store in sorted((world["data"] / "baselines" / "gefs").glob("202407*.zarr")):
        if store.name > "2024071712.zarr":
            shutil.rmtree(store)
            removed += 1
    assert removed == 28
    out = tmp_path / "experiments"
    assert run(world, out, periods=("2024-01", "2024-07")) == 0
    printed = capsys.readouterr().out
    assert "[2024-07] stops at 2024-07-17T12Z: the last 28 init time(s)" in printed
    assert "[2024-01] stops" not in printed
    cards = {m: json.loads((out / f"{m}_2024-07.json").read_text()) for m in MODELS}
    for model, card in cards.items():
        Scorecard.model_validate(card)
        assert card["n_init_times"] == 34, model
        assert (card["period_start"], card["period_end"]) == ("2024-07-01T00:00:00", "2024-07-17T12:00:00")
        assert card["init_hours_scored"] == [0, 12] and card["sealed_data_used"] is False
    assert cards["gefs"]["baseline_sha256"] == cards["climatology"]["forecast_sha256"]
    assert cards["persistence"]["baseline_sha256"] == cards["climatology"]["forecast_sha256"]
    # January, untouched by the rule, is byte-identical to a run with the unchanged config.
    reference = tmp_path / "reference"
    assert run(copy_world(base_world, tmp_path / "unchanged"), reference, periods=("2024-01",)) == 0
    for model in MODELS:
        assert (out / f"{model}_2024-01.json").read_bytes() == (
            reference / f"{model}_2024-01.json"
        ).read_bytes()
    # A rerun changes nothing.
    before = {p.name: p.read_bytes() for p in out.glob("*.json")}
    assert run(world, out, periods=("2024-01", "2024-07")) == 0
    assert {p.name: p.read_bytes() for p in out.glob("*.json")} == before

    # The summary tables are built from these files alone and show the shortened month.
    tables = run_gate1.summary_tables(out, 2024)
    assert [line for line in tables.splitlines() if line.startswith("### ")] == [
        *(f"### `{v}`" for v in ("t2m", "td2m", "u10", "v10", "msl")),
        "### Single leads at which GEFS scores below climatology",
    ]
    losing = tables.split("### Single leads")[1]
    expected_losing = sum(
        entry["crpss_vs_baseline"] < 0
        for month in ("2024-01", "2024-07")
        for per_lead in json.loads((out / f"gefs_{month}.json").read_text())["metrics"].values()
        for lead, entry in per_lead.items()
        if lead != "all"
    )
    assert sum(line.startswith("| 2024-") for line in losing.splitlines()) == expected_losing
    assert ("| none |" in losing) == (expected_losing == 0)
    t2m = tables.split("### `td2m`")[0].splitlines()
    july = next(line for line in t2m if line.startswith("| 2024-07 |"))
    cells = [c.strip() for c in july.strip("|").split("|")]
    entry = cards["gefs"]["metrics"]["t2m"]["all"]
    assert cells[1] == "34"
    assert cells[2] == f"{100 * entry['crpss_vs_baseline']:+.1f}"
    assert cells[3] == f"{100 * entry['crpss_vs_baseline_ecdf']:+.1f}"
    assert cells[4] == f"{entry['spread_skill_ratio']:.2f}"
    failed = [g for g in run_gate1.SUMMARY_GATES if not cards["gefs"]["gates"]["t2m"][g]]
    assert cells[5] == ("pass" if not failed else "fails " + ", ".join(failed))
    assert cells[8] == f"{cards['persistence']['metrics']['t2m']['all']['spread_skill_ratio']:.2f}"
    assert cells[10] == f"{cards['climatology']['metrics']['t2m']['all']['spread_skill_ratio']:.2f}"
    march = next(line for line in t2m if line.startswith("| 2024-03 |"))
    assert set(c.strip() for c in march.strip("|").split("|")[1:]) == {"n/a"}
    capsys.readouterr()
    assert run_gate1.main(["summary", "--scorecards", str(out), "--year", "2024"]) == 0
    assert capsys.readouterr().out.strip() == tables.strip()
    assert run_gate1.main(["summary", "--scorecards", str(out), "--year", "2023"]) == 2


def test_committed_summary_matches_the_committed_scorecards() -> None:
    """`experiments/SUMMARY_2024.md` must hold exactly what the summary command prints from the files."""
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    text = (experiments / "SUMMARY_2024.md").read_text()
    block = text.split("<!-- tables:start -->\n")[1].split("<!-- tables:end -->")[0]
    assert block.strip() == run_gate1.summary_tables(experiments, 2024).strip()
    # 36 scorecards, every one valid, none sealed, and the three models of a month share their coverage.
    for month in range(1, 13):
        cards = [json.loads((experiments / f"{m}_2024-{month:02d}.json").read_text()) for m in MODELS]
        for card in cards:
            Scorecard.model_validate(card)
            assert card["sealed_data_used"] is False and card["period_end"] < "2024-12-18"
        assert len({(c["period_start"], c["period_end"], c["n_init_times"]) for c in cards}) == 1
    assert json.loads((experiments / "gefs_2024-12.json").read_text())["period_end"] == "2024-12-17T12:00:00"


def test_quarter_periods_with_the_real_config() -> None:
    cfg = load_config()
    sizes = {}
    for number in range(1, 5):
        quarter = run_gate1.parse_period(f"2024-Q{number}", [0, 12], cfg)
        months = [run_gate1.parse_period(m, [0, 12], cfg) for m in quarter.months]
        assert quarter.months == tuple(f"2024-{m:02d}" for m in range(3 * number - 2, 3 * number + 1))
        # a quarter is exactly its three months, in time order, as one period
        assert quarter.init_times.tolist() == np.concatenate([m.init_times for m in months]).tolist()
        sizes[quarter.label] = (int(quarter.init_times.size), quarter.n_dropped)
    assert sizes == {"2024-Q1": (182, 0), "2024-Q2": (182, 0), "2024-Q3": (184, 0), "2024-Q4": (156, 28)}
    # Q4 ends where December does: the last init inside the validation split
    q4 = run_gate1.parse_period("2024-Q4", [0, 12], cfg)
    assert q4.init_times.max() == np.datetime64("2024-12-17T12", "ns") and q4.cycles[-1] == "2024121712"
    assert run_gate1.parse_period("2024-03", [0, 12], cfg).months == ()  # a month is not a quarter
    for label in ("2024-Q0", "2024-Q5", "2024-q1", "2024-Q", "2024-Q12", "24-Q1", "2025-Q1", "2023-Q4"):
        with pytest.raises(run_gate1.Refused):
            run_gate1.parse_period(label, [0, 12], cfg)


def test_runner_scores_a_quarter_as_one_period_from_its_month_stores(
    base_world: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture, no_network: None
) -> None:
    from wrfcomp.harness.hashing import content_sha256

    world = copy_world(base_world, tmp_path / "w")
    out = tmp_path / "experiments"
    # a quarter needs the stores of its months: asked for alone, every model fails and says what to run
    assert run(world, out, periods=("2024-Q1",)) == 1
    printed = capsys.readouterr().out
    assert (
        printed.count("FAILED") == 3
        and "run the months of 2024-Q1 (2024-01, 2024-02, 2024-03) first" in printed
    )
    assert not list(out.glob("*.json"))

    # GEFS cycles for February and March (the world has January and July); same truth as the world's
    truth, clean = make_truth("2023-12-31T00", "2024-08-03T00", seed=2)
    write_gefs_cycles(
        world["data"] / "baselines" / "gefs", truth, clean, seed=11, periods=("2024-02", "2024-03")
    )
    months = ("2024-01", "2024-02", "2024-03")
    assert run(world, out, periods=(*months, "2024-Q1")) == 0
    printed = capsys.readouterr().out
    assert "[2024-Q1] gefs: joined the stores of 2024-01, 2024-02, 2024-03" in printed

    work = world["data"] / "gate1"
    for model in MODELS:
        quarter = json.loads((out / f"{model}_2024-Q1.json").read_text())
        monthly = [json.loads((out / f"{model}_{m}.json").read_text()) for m in months]
        Scorecard.model_validate(quarter)
        assert quarter["n_init_times"] == sum(c["n_init_times"] for c in monthly) == 182
        assert (quarter["period_start"], quarter["period_end"]) == (
            "2024-01-01T00:00:00",
            "2024-03-31T12:00:00",
        )
        assert quarter["sealed_data_used"] is False and quarter["leads_scored"] == monthly[0]["leads_scored"]
        # one evaluation over the whole quarter: the same points as the three months together
        for variable, per_lead in quarter["metrics"].items():
            for lead, entry in per_lead.items():
                assert entry["n_pairs"] == sum(c["metrics"][variable][lead]["n_pairs"] for c in monthly)
        # the quarter's forecast is the month stores joined along init time, nothing else
        parts = [xr.open_zarr(work / f"{m}_h00-12" / f"{model}.zarr", chunks=None).load() for m in months]
        joined = xr.concat(parts, dim="init_time", data_vars="all", coords="minimal", compat="override")
        assert quarter["forecast_sha256"] == content_sha256(joined.sortby("init_time"))
        assert quarter["forecast_sha256"] not in {c["forecast_sha256"] for c in monthly}
    cards = {m: json.loads((out / f"{m}_2024-Q1.json").read_text()) for m in MODELS}
    assert cards["gefs"]["baseline_sha256"] == cards["climatology"]["forecast_sha256"]

    # deterministic: a rerun reuses the quarter stores and changes nothing
    before = {p.name: p.read_bytes() for p in out.glob("*.json")}
    assert run(world, out, periods=(*months, "2024-Q1")) == 0
    assert {p.name: p.read_bytes() for p in out.glob("*.json")} == before
    assert "[2024-Q1] climatology: forecast store present, reused" in capsys.readouterr().out


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


def test_quarter_assembly_checks_the_sealed_split_before_loading_a_month_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config()
    quarter = run_gate1.parse_period("2024-Q4", [0, 12], cfg)
    months = [run_gate1.parse_period(m, [0, 12], cfg) for m in quarter.months]
    # October's own init times, but with a lead that carries its valid times into 2025
    leads = np.array([48, 24 * 100], dtype="timedelta64[h]").astype("timedelta64[ns]")
    shape = (months[0].init_times.size, leads.size, 1, 1)
    store = xr.Dataset(
        {"t2m": (("init_time", "lead_time", "station_id", "ens_member"), np.zeros(shape, dtype="float32"))},
        coords={
            "init_time": months[0].init_times,
            "lead_time": leads,
            "station_id": ["S0"],
            "ens_member": [0],
        },
        attrs={"model_id": "noaa-gefs", "model_version": "1"},
    )
    store.to_zarr(tmp_path / "october.zarr", mode="w", consolidated=False)

    def no_values(self, **kwargs):
        raise AssertionError("a month store was loaded before the sealed-split check")

    monkeypatch.setattr(xr.Dataset, "load", no_values)
    with pytest.raises(run_gate1.ModelFailed, match="refused before it is read.*sealed split"):
        run_gate1.assemble_quarter(
            tmp_path / "q4.zarr",
            quarter,
            [tmp_path / "october.zarr"] * 3,
            months,
            "gefs",
            cfg.splits.sealed_test.start,
        )
    assert not (tmp_path / "q4.zarr").exists()


# --------------------------------------------------------------------------- confirmation period


def test_thresholds_train_end_cuts_the_input_and_records_the_cutoff(
    base_world: dict[str, Path], tmp_path: Path, no_network: None
) -> None:
    config = ["--config", str(base_world["config"])]
    whole = base_world["data"] / "truth" / "isd_conus_train.nc"  # 2022-12-15 .. 2023-08-15
    out = tmp_path / "cut.nc"
    args = ["--train-truth", str(whole), "--out", str(out), *config]
    assert build_thresholds.main([*args, "--train-end", "2023-03-31T23:00Z"]) == 0
    with xr.open_dataset(out) as cut, xr.open_dataset(whole) as train:
        cut.load()
        assert cut.attrs["train_end_utc"] == "2023-03-31T23:00:00.000000000"
        validate_thresholds(cut, variables=load_config().evaluation.variables.primary)
        # nothing after the cutoff is used: April to August have no threshold now
        assert np.isnan(cut["t2m"].sel(month=[4, 5, 6, 7, 8]).values).all()
        assert np.isfinite(cut["t2m"].sel(month=[1, 2, 3, 12]).values).all()
        # the same numbers as building from a store that was cut by hand
        by_hand_truth = train.sel(valid_time=slice(None, "2023-03-31T23"))
        by_hand_truth.to_netcdf(tmp_path / "hand.nc", engine="h5netcdf")
    hand = tmp_path / "hand_out.nc"
    hand_args = ["--train-truth", str(tmp_path / "hand.nc"), "--out", str(hand), *config]
    assert build_thresholds.main(hand_args) == 0
    with xr.open_dataset(out) as cut, xr.open_dataset(hand) as by_hand:
        for name in cut.data_vars:
            np.testing.assert_array_equal(cut[name].values, by_hand[name].values)
        # without the option the attribute is the train split end, whatever the data covers
        assert by_hand.attrs["train_end_utc"] == "2023-12-17T18:00:00.000000000"
    # one hour matters: the cutoff is inclusive
    with xr.open_dataset(whole) as train:
        march = train["t2m"].sel(valid_time=slice("2023-03-01T00", "2023-03-31T22")).values
    earlier = tmp_path / "earlier.nc"
    assert build_thresholds.main([*args[:3], str(earlier), *config, "--train-end", "2023-03-31T22"]) == 0
    with xr.open_dataset(earlier) as got:
        expected = np.nanquantile(march.astype(np.float64), 0.95, axis=0).astype(np.float32)
        np.testing.assert_array_equal(got["t2m"].sel(month=3).values, expected)


def test_thresholds_train_end_can_only_be_earlier(base_world: dict[str, Path], tmp_path: Path) -> None:
    config = ["--config", str(base_world["config"])]
    whole = str(base_world["data"] / "truth" / "isd_conus_train.nc")
    out = tmp_path / "x.nc"
    for bad in ("2023-12-17T18:00", "2023-12-17T19:00", "2024-06-01", "not-a-time", "NaT", "2020-01-01"):
        args = ["--train-truth", whole, "--out", str(out), *config, "--train-end", bad]
        assert build_thresholds.main(args) == 2
        assert not out.exists()
    # input past the train split is still refused; the option does not cut it away
    late = str(base_world["data"] / "truth" / "isd_conus_2024.nc")
    code = build_thresholds.main(
        ["--train-truth", whole, late, "--out", str(out), *config, "--train-end", "2023-03-31T23"]
    )
    assert code == 2
    assert not out.exists()


CONFIRM_CUTOFF = "2023-01-15T18"
CONFIRM_PERIOD = "2023-02"


def build_confirmation_world(root: Path) -> dict[str, Path]:
    """Train truth that contains the confirmation month, its GEFS cycles, and thresholds cut at the
    confirmation cutoff. The scored truth is the train truth file itself."""
    data = root / "data"
    (data / "truth").mkdir(parents=True)
    (data / "baselines" / "gefs").mkdir(parents=True)
    train, clean = make_truth("2022-01-15T00", "2023-03-05T00", seed=21)
    train.to_netcdf(data / "truth" / "isd_conus_train.nc", engine="h5netcdf")
    write_gefs_cycles(data / "baselines" / "gefs", train, clean, seed=22, periods=(CONFIRM_PERIOD,))
    config = write_config(root, list(station_ids(N_STATIONS)))
    world = {"root": root, "data": data, "config": config}
    for name, extra in (("tail_p95_confirm.nc", ["--train-end", CONFIRM_CUTOFF]), ("tail_p95.nc", [])):
        code = build_thresholds.main(
            [
                "--train-truth", str(data / "truth" / "isd_conus_train.nc"),
                "--out", str(data / "thresholds" / name),
                "--config", str(config),
                *extra,
            ]
        )  # fmt: skip
        assert code == 0
    return world


def confirmation_args(world: dict[str, Path], thresholds: str = "tail_p95_confirm.nc") -> list[str]:
    return [
        "--confirmation-train-end", CONFIRM_CUTOFF,
        "--truth", str(world["data"] / "truth" / "isd_conus_train.nc"),
        "--thresholds", str(world["data"] / "thresholds" / thresholds),
    ]  # fmt: skip


@pytest.fixture(scope="module")
def confirmation_world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return build_confirmation_world(tmp_path_factory.mktemp("confirm"))


def _store_hashes(world: dict[str, Path]) -> dict[str, str]:
    from wrfcomp.harness.hashing import content_sha256

    out = {}
    for model in MODELS:
        with xr.open_zarr(world["data"] / "gate1" / f"{CONFIRM_PERIOD}_h00-12" / f"{model}.zarr") as ds:
            out[model] = content_sha256(ds)
    return out


def test_confirmation_mode_scores_a_train_split_month_with_cut_references(
    confirmation_world: dict[str, Path], tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture
) -> None:
    world = copy_world(confirmation_world, tmp_path / "w")
    out = tmp_path / "cards"
    # without the mode a period before the validation split is refused, as before
    assert run(world, out, periods=(CONFIRM_PERIOD,)) == 2
    assert run(world, out, *confirmation_args(world), periods=(CONFIRM_PERIOD,)) == 0
    assert "confirmation mode" in capsys.readouterr().out
    cutoff = str(np.datetime64(CONFIRM_CUTOFF, "ns"))
    work = world["data"] / "gate1" / f"{CONFIRM_PERIOD}_h00-12"
    for model in MODELS:
        card = json.loads((out / f"{model}_{CONFIRM_PERIOD}.json").read_text())
        Scorecard.model_validate(card)
        assert card["period_start"] == "2023-02-01T00:00:00"
        assert card["period_end"] == "2023-02-28T12:00:00"
        assert card["n_init_times"] == 56
        assert card["sealed_data_used"] is False
        with xr.open_zarr(work / f"{model}.zarr") as ds:
            assert ds.attrs.get("train_end_utc") == (None if model == "gefs" else cutoff)
    clean = _store_hashes(world)

    # The cutoff is obeyed: change every train value between the cutoff and the period (kept clear of
    # the last hours before the first init, which persistence starts from). The references and their
    # scorecards must not move.
    poisoned = copy_world(confirmation_world, tmp_path / "p")
    path = poisoned["data"] / "truth" / "isd_conus_train.nc"
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    for name in VARIABLES:
        ds[name].loc[dict(valid_time=slice("2023-01-15T19", "2023-01-31T12"))] += 50.0
    ds.to_netcdf(path, engine="h5netcdf")
    out_p = tmp_path / "cards_p"
    assert run(poisoned, out_p, *confirmation_args(poisoned), periods=(CONFIRM_PERIOD,)) == 0
    assert _store_hashes(poisoned) == clean
    for model in MODELS:
        a = json.loads((out / f"{model}_{CONFIRM_PERIOD}.json").read_text())
        b = json.loads((out_p / f"{model}_{CONFIRM_PERIOD}.json").read_text())
        assert a["truth_sha256"] != b["truth_sha256"]  # the file did change
        assert a["forecast_sha256"] == b["forecast_sha256"]
        assert a["metrics"] == b["metrics"] and a["n_pairs"] == b["n_pairs"]

    # a store built with another cutoff, or with none, is not reused
    period = run_gate1.parse_period(
        CONFIRM_PERIOD, [0, 12], load_config(), np.datetime64(CONFIRM_CUTOFF, "ns")
    )
    stations = list(station_ids(N_STATIONS))
    for other in (None, np.datetime64("2023-01-10T00", "ns")):
        with pytest.raises(run_gate1.ModelFailed, match="training cutoff"):
            run_gate1.existing_store_ok(work / "climatology.zarr", period, stations, "climatology", other)
    assert run_gate1.existing_store_ok(work / "gefs.zarr", period, stations, "gefs", period.init_times[0])


def test_confirmation_mode_refusals(
    confirmation_world: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    world = confirmation_world  # nothing is written to the data dir by a refused run
    out = tmp_path / "cards"
    full = confirmation_args(world)

    def refused(*extra: str, periods: tuple[str, ...] = (CONFIRM_PERIOD,)) -> str:
        capsys.readouterr()
        assert run(world, out, *extra, periods=periods) == 2
        return capsys.readouterr().err

    # the defaults are the validation truth and the whole-train thresholds: both must be named
    assert "needs --truth and --thresholds" in refused("--confirmation-train-end", CONFIRM_CUTOFF)
    assert "needs --truth and --thresholds" in refused(*full[:4])
    # thresholds of the whole train split would leak the period into the tail thresholds
    assert "not up to the confirmation cutoff" in refused(*confirmation_args(world, "tail_p95.nc"))
    # the cutoff can only be earlier than the train split end
    for late in ("2023-12-17T18", "2024-03-01"):
        assert "can only be earlier" in refused("--confirmation-train-end", late, *full[2:])
    assert "not a time" in refused("--confirmation-train-end", "soon", *full[2:])
    # embargo: the period must start 14 days after the cutoff
    assert "days after the confirmation cutoff" in refused(*full, periods=("2023-01",))
    # a validation period is not a confirmation period
    assert "reaches the validation split" in refused(*full, periods=("2024-01",))
    assert not (world["data"] / "gate1").exists()
    assert not out.exists() or not list(out.iterdir())


def test_confirmation_periods_of_2023_with_the_real_config() -> None:
    """The proposed 2023 confirmation year: cutoff 2022-12-17T18Z, last init 2023-12-15T12Z."""
    cfg = load_config()
    cutoff = np.datetime64("2022-12-17T18", "ns")
    quarters = [run_gate1.parse_period(f"2023-Q{q}", [0, 12], cfg, cutoff) for q in range(1, 5)]
    assert [p.init_times.size for p in quarters] == [180, 182, 184, 152]
    assert quarters[0].init_times.min() == np.datetime64("2023-01-01T00", "ns")
    assert quarters[3].init_times.max() == np.datetime64("2023-12-15T12", "ns")
    assert quarters[3].n_dropped == 32
    assert quarters[3].months == ("2023-10", "2023-11", "2023-12")
    december = run_gate1.parse_period("2023-12", [0, 12], cfg, cutoff)
    assert december.init_times.size == 30 and december.n_dropped == 32
    # one day closer to the period and the 14-day embargo is broken
    with pytest.raises(run_gate1.Refused, match="days after the confirmation cutoff"):
        run_gate1.parse_period("2023-Q1", [0, 12], cfg, np.datetime64("2022-12-18T06", "ns"))
    # the ordinary rule is untouched
    with pytest.raises(run_gate1.Refused, match="before the validation split"):
        run_gate1.parse_period("2023-Q1", [0, 12], cfg)
    assert run_gate1.parse_period("2024-Q4", [0, 12], cfg).init_times.size == 156


def test_slice_truth_copies_a_time_range_unchanged(
    confirmation_world: dict[str, Path], tmp_path: Path
) -> None:
    slice_truth = load_script("slice_truth")
    source = confirmation_world["data"] / "truth" / "isd_conus_train.nc"
    out = tmp_path / "cut.nc"
    args = ["--source", str(source), "--out", str(out), "--config", str(confirmation_world["config"])]
    assert slice_truth.main([*args, "--start", "2023-01-31T18Z", "--end", "2023-03-02T12"]) == 0
    with xr.open_dataset(source) as whole, xr.open_dataset(out) as cut:
        want = whole.sel(valid_time=slice("2023-01-31T18", "2023-03-02T12")).load()
        xr.testing.assert_identical(cut.load(), want)
        assert cut["t2m_offset_min"].dtype == np.int16 and cut["t2m"].dtype == np.float32
    before = out.read_bytes()
    assert slice_truth.main([*args, "--start", "2023-01-31T18", "--end", "2023-03-02T12"]) == 2  # exists
    assert out.read_bytes() == before
    other = ["--source", str(source), "--out", str(tmp_path / "x.nc")]
    assert slice_truth.main([*other, "--start", "2030-01-01", "--end", "2030-02-01"]) == 2  # nothing there
    assert slice_truth.main([*other, "--start", "2023-03-01", "--end", "2023-02-01"]) == 2
    assert not (tmp_path / "x.nc").exists()
