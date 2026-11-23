from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from conftest import Synthetic, make_synthetic, open_zarr, station_ids, write_config, write_netcdf, write_zarr
from typer.testing import CliRunner

from wrfcomp.harness.align import align_truth
from wrfcomp.harness.evaluate import app, canonical_json
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.harness.schemas import Scorecard

HOUR = np.timedelta64(1, "h")
N_REQUIRED = 20  # stations in the synthetic stations file

# =================================================================== content hash


def test_hash_is_stable_and_hex(synthetic) -> None:
    first = content_sha256(synthetic.candidate)
    assert len(first) == 64 and int(first, 16) >= 0
    assert first == content_sha256(synthetic.candidate.copy(deep=True))


def test_same_data_different_chunking_same_hash(tmp_path, synthetic) -> None:
    ds = synthetic.candidate
    a = write_zarr(ds, tmp_path / "a.zarr", {"init_time": 30, "station_id": 20, "ens_member": 8})
    b = write_zarr(
        ds, tmp_path / "b.zarr", {"init_time": 7, "lead_time": 1, "station_id": 3, "ens_member": 5}
    )
    assert zarr.open_group(a)["t2m"].chunks != zarr.open_group(b)["t2m"].chunks
    expected = content_sha256(ds)
    assert content_sha256(open_zarr(a)) == expected
    assert content_sha256(open_zarr(b)) == expected
    assert content_sha256(xr.open_zarr(b, consolidated=False)) == expected  # dask-backed


def test_zarr_and_netcdf_of_the_same_truth_hash_the_same(tmp_path, synthetic) -> None:
    expected = content_sha256(synthetic.truth)
    z = write_zarr(synthetic.truth, tmp_path / "t.zarr", {"valid_time": 50, "station_id": 4})
    assert content_sha256(open_zarr(z)) == expected
    for netcdf3 in (False, True):
        n = write_netcdf(synthetic.truth, tmp_path / f"t{int(netcdf3)}.nc", netcdf3=netcdf3)
        assert content_sha256(xr.open_dataset(n)) == expected


def test_one_value_changed_changes_hash(synthetic) -> None:
    before = content_sha256(synthetic.candidate)
    changed = synthetic.candidate.copy(deep=True)
    value = changed["u10"].values[17, 2, 11, 6]
    changed["u10"].values[17, 2, 11, 6] = np.nextafter(value, np.float32(np.inf))
    assert content_sha256(changed) != before


def test_coordinate_changes_change_hash(synthetic) -> None:
    before = content_sha256(synthetic.candidate)
    lat = synthetic.candidate.copy(deep=True)
    lat["latitude"].values[3] += 0.01
    shifted = synthetic.candidate.assign_coords(
        init_time=synthetic.candidate["init_time"].values + np.timedelta64(1, "h")
    )
    station = synthetic.candidate.copy(deep=True)
    ids = station["station_id"].values.copy()
    ids[0] = "70000099998"
    station = station.assign_coords(station_id=ids)
    assert len({before, content_sha256(lat), content_sha256(shifted), content_sha256(station)}) == 4


def test_names_dims_and_dtypes_are_covered(synthetic) -> None:
    ds = synthetic.thresholds
    before = content_sha256(ds)
    assert content_sha256(ds.rename({"t2m": "td2m"})) != before
    assert content_sha256(ds.astype("float64")) != before
    assert content_sha256(ds.transpose("month", "station_id")) != before
    assert content_sha256(ds.drop_vars("u10")) != before


def test_attributes_and_insertion_order_are_not_covered(synthetic) -> None:
    ds = synthetic.thresholds
    before = content_sha256(ds)
    reordered = xr.Dataset({name: ds[name] for name in reversed(list(ds.data_vars))})
    assert list(reordered.data_vars) != list(ds.data_vars)
    assert content_sha256(reordered) == before
    noted = ds.copy()
    noted.attrs = {"b": 1, "a": 2}
    noted["t2m"].attrs = {"units": "K", "note": "x"}
    assert content_sha256(noted) == before


def test_string_storage_does_not_matter() -> None:
    values = np.arange(6, dtype="float32").reshape(2, 3)
    ids = ["72494023234", "72295023174", "74486094789"]
    hashes = {
        content_sha256(xr.Dataset({"t2m": (("month", "station_id"), values)}, coords={"station_id": coord}))
        for coord in (np.array(ids, dtype=object), np.array(ids, dtype="U11"), np.array(ids, dtype="U20"))
    }
    assert len(hashes) == 1


def test_nan_positions_are_covered_and_empty_is_fine() -> None:
    a = xr.Dataset({"t2m": ("x", np.array([1.0, np.nan, 3.0], "float32"))})
    b = xr.Dataset({"t2m": ("x", np.array([1.0, 3.0, np.nan], "float32"))})
    assert content_sha256(a) != content_sha256(b)
    assert content_sha256(a) == content_sha256(a.copy(deep=True))
    assert len(content_sha256(xr.Dataset())) == 64


# =================================================================== wrfcomp-eval


def _stores(tmp_path: Path, syn: Synthetic, *, chunks: dict[str, int] | None = None) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return {
        "forecast": write_zarr(syn.candidate, tmp_path / "candidate.zarr", chunks),
        "baseline": write_zarr(syn.baseline, tmp_path / "baseline.zarr", chunks),
        "truth": write_zarr(syn.truth, tmp_path / "truth.zarr"),
        "thresholds": write_netcdf(syn.thresholds, tmp_path / "thresholds.nc"),
        "config": write_config(tmp_path, list(station_ids(N_REQUIRED))),
    }


def _run(stores: dict[str, Path], out: Path, *extra: str, baseline: bool = True, model_id: str = "synth"):
    args = ["--forecast", str(stores["forecast"]), "--truth", str(stores["truth"])]
    args += ["--thresholds", str(stores["thresholds"]), "--out", str(out)]
    args += ["--model-id", model_id, "--model-version", "1", "--config", str(stores["config"])]
    if baseline:
        args += ["--baseline", str(stores["baseline"]), "--baseline-id", "synth_base"]
    return CliRunner().invoke(app, [*args, *extra])


def _scored(result, out: Path) -> dict:
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    return json.loads(out.read_text())


def test_end_to_end_scorecard_validates_and_is_byte_identical(tmp_path, synthetic) -> None:
    stores = _stores(tmp_path / "a", synthetic)
    card = _scored(_run(stores, tmp_path / "s1.json"), tmp_path / "s1.json")
    first = (tmp_path / "s1.json").read_bytes()

    model = Scorecard.model_validate(card)
    assert model.model_id == "synth" and model.baseline_id == "synth_base"
    assert (model.n_stations, model.n_init_times) == (20, 30)
    assert model.leads_scored == [6, 12, 24] and model.init_hours_scored == [0, 12]
    assert set(model.metrics) == {"t2m", "u10"}
    assert model.forecast_sha256 == content_sha256(synthetic.candidate)
    assert model.truth_sha256 == content_sha256(synthetic.truth)
    assert model.baseline_sha256 == content_sha256(synthetic.baseline)
    for per_lead in card["metrics"].values():
        assert set(per_lead) == {"all", "6", "12", "24"}
        assert per_lead["all"]["n_pairs"] == sum(per_lead[k]["n_pairs"] for k in ("6", "12", "24"))
        assert per_lead["all"]["crps"] > 0

    # Same inputs again: same bytes.
    _scored(_run(stores, tmp_path / "s2.json"), tmp_path / "s2.json")
    assert (tmp_path / "s2.json").read_bytes() == first
    # Same data stored with other chunks, in another directory: still the same bytes.
    other = _stores(tmp_path / "b", synthetic, chunks={"init_time": 7, "station_id": 3, "ens_member": 5})
    _scored(_run(other, tmp_path / "s3.json"), tmp_path / "s3.json")
    assert (tmp_path / "s3.json").read_bytes() == first

    text = first.decode("ascii")
    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert str(tmp_path) not in text and "zarr" not in text
    assert text == canonical_json(json.loads(text))  # sorted keys, canonical floats


@pytest.mark.parametrize("netcdf3", [False, True], ids=["netcdf4", "netcdf3"])
def test_truth_and_thresholds_as_netcdf_score_the_same(tmp_path, synthetic, netcdf3: bool) -> None:
    stores = _stores(tmp_path, synthetic)
    a = _scored(_run(stores, tmp_path / "a.json"), tmp_path / "a.json")
    stores["truth"] = write_netcdf(synthetic.truth, tmp_path / "truth.nc", netcdf3=netcdf3)
    stores["thresholds"] = write_netcdf(synthetic.thresholds, tmp_path / "th.nc", netcdf3=netcdf3)
    magic = stores["truth"].read_bytes()[:4]
    assert magic.startswith(b"CDF") if netcdf3 else magic == b"\x89HDF"
    b = _scored(_run(stores, tmp_path / "b.json"), tmp_path / "b.json")
    assert a["truth_sha256"] == b["truth_sha256"] == content_sha256(synthetic.truth)
    assert a["thresholds_sha256"] == content_sha256(synthetic.thresholds)
    if netcdf3:
        # NetCDF3 has no 64-bit integers, so `month` comes back as int32. The hash covers dtypes, so the
        # thresholds hash differs; nothing else in the scorecard does.
        assert xr.open_dataset(stores["thresholds"])["month"].dtype == np.dtype("int32")
        assert b.pop("thresholds_sha256") != a.pop("thresholds_sha256")
        assert a == b
    else:
        assert (tmp_path / "b.json").read_bytes() == (tmp_path / "a.json").read_bytes()


def test_forecast_nan_at_valid_truth_point_exits_3(tmp_path, synthetic) -> None:
    obs = align_truth(synthetic.candidate, synthetic.truth)
    assert np.isfinite(obs["t2m"].values[11, 2, 5])
    synthetic.candidate["t2m"].values[11, 2, 5, 0] = np.nan
    out = tmp_path / "s.json"
    result = _run(_stores(tmp_path, synthetic), out)
    assert result.exit_code == 3, result.output
    init = np.datetime_as_string(synthetic.candidate["init_time"].values[11], unit="s") + "Z"
    assert f"(t2m, {init}, 24h, {synthetic.candidate['station_id'].values[5]})" in result.output
    assert "candidate" in result.output
    assert not out.exists()


def test_baseline_nan_at_valid_truth_point_exits_3(tmp_path, synthetic) -> None:
    obs = align_truth(synthetic.candidate, synthetic.truth)
    assert np.isfinite(obs["u10"].values[0, 0, 0])
    synthetic.baseline["u10"].values[0, 0, 0, 7] = np.nan
    result = _run(_stores(tmp_path, synthetic), tmp_path / "s.json")
    assert result.exit_code == 3, result.output
    assert "baseline" in result.output


def test_truth_nan_point_is_skipped_for_candidate_and_baseline_alike(tmp_path, synthetic) -> None:
    obs = align_truth(synthetic.candidate, synthetic.truth)
    reference = _scored(_run(_stores(tmp_path / "a", synthetic), tmp_path / "a.json"), tmp_path / "a.json")
    for name in ("t2m", "u10"):
        holes = np.argwhere(np.isnan(obs[name].values))
        assert len(holes) > 0, "fixture should have truth gaps on the forecast grid"
        assert reference["n_pairs"][name] == obs[name].size - len(holes)
        # What the forecasts say where truth is missing must not matter: absurd in one, NaN in the other.
        for i, j, k in holes:
            synthetic.candidate[name].values[i, j, k, :] = 1.0e6
            synthetic.baseline[name].values[i, j, k, :] = np.nan
    changed = _scored(_run(_stores(tmp_path / "b", synthetic), tmp_path / "b.json"), tmp_path / "b.json")
    assert changed["forecast_sha256"] != reference["forecast_sha256"]
    assert changed["metrics"] == reference["metrics"]
    assert changed["n_pairs"] == reference["n_pairs"]

    # One more truth value removed: exactly one pair fewer, for the candidate and the baseline together.
    i, j, k = np.argwhere(np.isfinite(obs["t2m"].values))[0]
    when = synthetic.candidate["init_time"].values[i] + synthetic.candidate["lead_time"].values[j]
    synthetic.truth["t2m"].loc[
        {"valid_time": when, "station_id": synthetic.truth["station_id"].values[k]}
    ] = np.nan
    fewer = _scored(_run(_stores(tmp_path / "c", synthetic), tmp_path / "c.json"), tmp_path / "c.json")
    assert fewer["n_pairs"]["t2m"] == reference["n_pairs"]["t2m"] - 1
    assert fewer["n_pairs"]["u10"] == reference["n_pairs"]["u10"]


def test_2025_init_time_is_refused_without_allow_sealed(tmp_path) -> None:
    syn = make_synthetic(start="2024-12-20T00:00")
    assert syn.candidate["init_time"].values.max() >= np.datetime64("2025-01-01")
    out = tmp_path / "s.json"
    result = _run(_stores(tmp_path, syn), out)
    assert result.exit_code == 2, result.output
    assert "sealed" in result.output
    assert not out.exists()


def test_2024_init_with_2025_valid_time_is_refused(tmp_path) -> None:
    syn = make_synthetic(start="2024-12-17T00:00")  # last init 2024-12-31T12, +24 h is in 2025
    assert syn.candidate["init_time"].values.max() == np.datetime64("2024-12-31T12")
    # Truth cut to 2024 so that only the forecast's valid time can be the reason.
    syn.truth = syn.truth.sel(valid_time=slice(None, "2024-12-31T23"))
    result = _run(_stores(tmp_path, syn), tmp_path / "s.json")
    assert result.exit_code == 2, result.output
    assert "valid_time" in result.output and "sealed" in result.output


def test_allow_sealed_scores_with_a_loud_warning(tmp_path) -> None:
    syn = make_synthetic(start="2024-12-20T00:00")
    out = tmp_path / "s.json"
    result = _run(_stores(tmp_path, syn), out, "--allow-sealed")
    card = _scored(result, out)
    Scorecard.model_validate(card)
    assert card["sealed_data_used"] is True
    assert card["period_end"] == "2025-01-03T12:00:00"
    assert "WARNING" in result.output and "SEALED SPLIT IN USE" in result.output


def test_candidate_equal_to_baseline_has_zero_skill_and_fails_the_gate(tmp_path, synthetic) -> None:
    stores = _stores(tmp_path, synthetic)
    stores["baseline"] = stores["forecast"]
    out = tmp_path / "s.json"
    card = _scored(_run(stores, out), out)
    assert card["baseline_sha256"] == card["forecast_sha256"]
    for name, per_lead in card["metrics"].items():
        for lead, entry in per_lead.items():
            assert entry["crpss_vs_baseline"] == 0.0, (name, lead)
        assert card["gates"][name]["skill"] is False
    assert card["passed_gate"] is False


def test_no_baseline_scores_but_cannot_pass(tmp_path, synthetic) -> None:
    out = tmp_path / "s.json"
    card = _scored(_run(_stores(tmp_path, synthetic), out, baseline=False), out)
    assert card["baseline_id"] is None and card["baseline_sha256"] is None
    assert card["passed_gate"] is False
    assert card["metrics"]["t2m"]["all"]["crpss_vs_baseline"] is None
    with_baseline = _scored(_run(_stores(tmp_path, synthetic), out), out)
    for key in ("crps", "twcrps_p95", "loss", "spread_skill_ratio", "n_pairs"):
        assert card["metrics"]["t2m"]["12"][key] == with_baseline["metrics"]["t2m"]["12"][key]


def test_better_candidate_passes_the_gate(tmp_path, synthetic) -> None:
    out = tmp_path / "s.json"
    card = _scored(_run(_stores(tmp_path, synthetic), out), out)
    assert card["gates"] == {
        v: {"skill": True, "tail": True, "spread_skill": True, "dm": True} for v in card["gates"]
    }
    assert card["passed_gate"] is True


def test_whole_missing_lead_is_allowed_and_recorded(tmp_path, synthetic) -> None:
    synthetic.candidate = synthetic.candidate.isel(lead_time=[0, 2])
    out = tmp_path / "s.json"
    card = _scored(_run(_stores(tmp_path, synthetic), out), out)
    assert card["leads_scored"] == [6, 24]
    assert set(card["metrics"]["t2m"]) == {"all", "6", "24"}


def test_dropped_station_is_refused(tmp_path, synthetic) -> None:
    gone = [str(s) for s in synthetic.candidate["station_id"].values[[3, 17]]]
    synthetic.candidate = synthetic.candidate.drop_sel(station_id=gone)
    out = tmp_path / "s.json"
    result = _run(_stores(tmp_path, synthetic), out)
    assert result.exit_code == 3, f"{result.output}\n{result.exception!r}"
    assert "candidate forecast is missing 2 required station(s)" in result.output
    assert all(station in result.output for station in gone)
    assert not out.exists()


def test_dropped_date_is_refused(tmp_path, synthetic) -> None:
    gone = synthetic.candidate["init_time"].values[[9, 10, 21]]
    synthetic.candidate = synthetic.candidate.drop_sel(init_time=gone)
    out = tmp_path / "s.json"
    result = _run(_stores(tmp_path, synthetic), out)
    assert result.exit_code == 3, f"{result.output}\n{result.exception!r}"
    assert "candidate forecast is missing 3 required init time(s)" in result.output
    for when in gone:
        assert np.datetime_as_string(when, unit="s") + "Z" in result.output
    assert not out.exists()


def test_dropping_one_init_hour_on_one_day_is_refused(tmp_path, synthetic) -> None:
    # 00Z and 12Z every day, except one 12Z: the cadence is still {00, 12}, so that 12Z is a missing date.
    gone = synthetic.candidate["init_time"].values[5]
    synthetic.candidate = synthetic.candidate.drop_sel(init_time=[gone])
    result = _run(_stores(tmp_path, synthetic), tmp_path / "s.json")
    assert result.exit_code == 3, result.output
    assert np.datetime_as_string(gone, unit="s") + "Z" in result.output


def test_more_than_ten_missing_items_lists_ten(tmp_path) -> None:
    syn = make_synthetic(n_init=40)
    syn.candidate = syn.candidate.isel(init_time=[0, 1, 2, 3, 36, 37, 38, 39])
    result = _run(_stores(tmp_path, syn), tmp_path / "s.json", baseline=False)
    assert result.exit_code == 3, result.output
    assert "missing 32 required init time(s); first 10:" in result.output
    assert result.output.count("T00:00:00Z") + result.output.count("T12:00:00Z") == 10


@pytest.mark.parametrize("what", ["station", "date"])
def test_baseline_that_lacks_a_candidate_point_is_refused(tmp_path, synthetic, what: str) -> None:
    if what == "station":
        synthetic.baseline = synthetic.baseline.isel(station_id=slice(0, 15))
    else:
        synthetic.baseline = synthetic.baseline.isel(init_time=slice(4, None))
    out = tmp_path / "s.json"
    result = _run(_stores(tmp_path, synthetic), out)
    assert result.exit_code == 3, f"{result.output}\n{result.exception!r}"
    assert "baseline forecast is missing" in result.output
    assert not out.exists()


def test_baseline_may_cover_more_than_the_candidate(tmp_path) -> None:
    syn = make_synthetic(n_init=30)
    reference = _scored(_run(_stores(tmp_path / "a", syn), tmp_path / "a.json"), tmp_path / "a.json")
    # Candidate covers the last 20 init times only; the baseline still has all 30.
    syn.candidate = syn.candidate.isel(init_time=slice(10, None))
    out = tmp_path / "b.json"
    card = _scored(_run(_stores(tmp_path / "b", syn), out), out)
    assert card["n_init_times"] == 20 and reference["n_init_times"] == 30
    assert card["period_start"] == "2024-03-06T00:00:00" and card["period_end"] == reference["period_end"]
    assert card["baseline_sha256"] == reference["baseline_sha256"]


def test_stations_outside_the_stations_file_are_not_scored(tmp_path) -> None:
    syn = make_synthetic(n_stations=23, n_init=12)
    out = tmp_path / "s.json"
    result = _run(_stores(tmp_path, syn), out)
    card = _scored(result, out)
    assert card["n_stations"] == N_REQUIRED
    assert "3 forecast station(s) are not in the stations file" in result.output
    obs = align_truth(syn.candidate.isel(station_id=slice(0, N_REQUIRED)), syn.truth)
    assert card["n_pairs"]["t2m"] == int(np.isfinite(obs["t2m"].values).sum())


def test_period_and_sealed_flag_are_recorded(tmp_path, synthetic) -> None:
    out = tmp_path / "s.json"
    card = _scored(_run(_stores(tmp_path, synthetic), out), out)
    assert card["period_start"] == "2024-03-01T00:00:00"
    assert card["period_end"] == "2024-03-15T12:00:00"
    assert card["init_hours_scored"] == [0, 12]
    assert card["sealed_data_used"] is False


def test_non_v1_lead_is_not_scored(tmp_path) -> None:
    syn = make_synthetic(lead_hours=(6, 9, 12), n_init=12)
    out = tmp_path / "s.json"
    result = _run(_stores(tmp_path, syn), out)
    card = _scored(result, out)
    assert card["leads_scored"] == [6, 12]
    assert "[9]" in result.output


CASES = ["missing", "model_id", "baseline_id", "variable", "members", "init_hour", "nan_threshold"]


@pytest.mark.parametrize("case", CASES)
def test_bad_input_exits_2(tmp_path, case: str) -> None:
    syn = make_synthetic(n_init=6)
    extra: tuple[str, ...] = ()
    kwargs: dict = {}
    if case == "variable":
        syn.truth = syn.truth.drop_vars("u10")
    elif case == "members":
        syn.candidate = syn.candidate.isel(ens_member=[0])
    elif case == "init_hour":
        syn.candidate = syn.candidate.assign_coords(init_time=syn.candidate["init_time"].values + HOUR * 3)
    elif case == "nan_threshold":
        syn.thresholds["t2m"].values[2, :] = np.nan
    stores = _stores(tmp_path, syn)
    if case == "missing":
        stores["forecast"] = tmp_path / "nope.zarr"
    elif case == "model_id":
        kwargs["model_id"] = "someone_else"
    elif case == "baseline_id":
        kwargs["baseline"] = False
        extra = ("--baseline-id", "synth_base")
    out = tmp_path / "s.json"
    result = _run(stores, out, *extra, **kwargs)
    assert result.exit_code == 2, f"{result.output}\n{result.exception!r}"
    assert "bad input" in result.output
    assert not out.exists()


def test_canonical_json_float_rule() -> None:
    text = canonical_json(
        {
            "b": 0.1 + 0.2,
            "a": 1 / 3,
            "n": float("nan"),
            "i": np.int64(7),
            "f": np.float32(2.5),
            "t": True,
            "z": None,
        }
    )
    assert json.loads(text) == {
        "a": 0.3333333333,
        "b": 0.3,
        "f": 2.5,
        "i": 7,
        "n": None,
        "t": True,
        "z": None,
    }
    assert text.index('"a"') < text.index('"b"') and text.endswith("}\n")
    assert '"i": 7,' in text and '"t": true' in text
    with pytest.raises(TypeError):
        canonical_json({"when": np.datetime64("2024-01-01")})
