"""Tests for scripts/build_scoring_bundle.py and scripts/fetch_scoring_data.py.

Uses only synthetic seeded datasets and local fake archives (never touches the network).
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wrfcomp.harness.hashing import content_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "splits.yaml"


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


builder = _load_script("build_scoring_bundle")
fetcher = _load_script("fetch_scoring_data")


def _make_truth(seed: int = 1) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01T00:00:00", periods=4, freq="1h").to_numpy(dtype="datetime64[ns]")
    stations = np.array(["72405013743", "72295023174"], dtype="U11")
    return xr.Dataset(
        {
            "t2m": (
                ("valid_time", "station_id"),
                rng.normal(280.0, 2.0, (4, 2)).astype(np.float32),
                {"units": "K"},
            ),
        },
        coords={
            "valid_time": times,
            "station_id": stations,
            "latitude": (
                "station_id",
                np.array([38.85, 33.94], dtype=np.float32),
                {"units": "degrees_north"},
            ),
            "longitude": (
                "station_id",
                np.array([-77.04, -118.41], dtype=np.float32),
                {"units": "degrees_east"},
            ),
            "elevation_m": ("station_id", np.array([4.0, 38.0], dtype=np.float32), {"units": "m"}),
        },
    )


def _make_thresholds(seed: int = 2) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    stations = np.array(["72405013743", "72295023174"], dtype="U11")
    return xr.Dataset(
        {
            "t2m": (
                ("station_id", "month"),
                rng.uniform(290.0, 305.0, (2, 12)).astype(np.float32),
                {"units": "K", "tail_quantile": 0.95, "train_end_utc": "2023-12-17T18:00:00.000000000"},
            ),
        },
        coords={"station_id": stations, "month": np.arange(1, 13, dtype=np.int32)},
        attrs={
            "protocol_version": "0.1.0",
            "tail_quantile": 0.95,
            "train_end_utc": "2023-12-17T18:00:00.000000000",
        },
    )


def _make_forecast(model_id: str, start: str, seed: int) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    inits = pd.date_range(start, periods=2, freq="12h").to_numpy(dtype="datetime64[ns]")
    leads = np.array([3, 6], dtype="timedelta64[h]").astype("timedelta64[ns]")
    stations = np.array(["72405013743", "72295023174"], dtype="U11")
    members = np.array([0, 1], dtype=np.int32)
    return xr.Dataset(
        {
            "t2m": (
                ("init_time", "lead_time", "station_id", "ens_member"),
                rng.normal(280.0, 1.5, (2, 2, 2, 2)).astype(np.float32),
                {"units": "K"},
            )
        },
        coords={
            "init_time": inits,
            "lead_time": leads,
            "station_id": stations,
            "ens_member": members,
            "latitude": (
                "station_id",
                np.array([38.85, 33.94], dtype=np.float32),
                {"units": "degrees_north"},
            ),
            "longitude": (
                "station_id",
                np.array([-77.04, -118.41], dtype=np.float32),
                {"units": "degrees_east"},
            ),
            "elevation_m": ("station_id", np.array([4.0, 38.0], dtype=np.float32), {"units": "m"}),
        },
        attrs={"model_id": model_id, "model_version": "test-1"},
    )


@pytest.fixture()
def synthetic_env(tmp_path: Path) -> dict[str, Any]:
    data_dir = tmp_path / "data"
    (data_dir / "truth").mkdir(parents=True)
    (data_dir / "thresholds").mkdir(parents=True)
    (data_dir / "gate1" / "2024-01_h00-12").mkdir(parents=True)
    (data_dir / "gate1" / "2024-Q1_h00-12").mkdir(parents=True)
    # Decoy private/unpinned files that must NEVER be bundled.
    (data_dir / "citizen").mkdir(parents=True)
    (data_dir / "citizen" / ".station_key").write_text("secret", encoding="utf-8")
    (data_dir / "citizen" / "seattle.nc").write_bytes(b"private-citizen-data")

    truth_ds = _make_truth(seed=10)
    truth_path = data_dir / "truth" / "isd_conus_2024.nc"
    truth_ds.to_netcdf(truth_path, engine="h5netcdf")
    with xr.open_dataset(truth_path) as opened:
        truth_sha = content_sha256(opened)

    thresh_ds = _make_thresholds(seed=11)
    thresh_path = data_dir / "thresholds" / "tail_p95.nc"
    thresh_ds.to_netcdf(thresh_path, engine="h5netcdf")
    with xr.open_dataset(thresh_path) as opened:
        thresh_sha = content_sha256(opened)

    stores = {
        ("2024-01", "climatology"): ("gate1/2024-01_h00-12/climatology.zarr", 20),
        ("2024-01", "noaa-gefs"): ("gate1/2024-01_h00-12/gefs.zarr", 21),
        ("2024-Q1", "climatology"): ("gate1/2024-Q1_h00-12/climatology.zarr", 22),
        ("2024-Q1", "noaa-gefs"): ("gate1/2024-Q1_h00-12/gefs.zarr", 23),
    }
    store_hashes: dict[tuple[str, str], str] = {}
    for (period, model), (rel, seed) in stores.items():
        ds = _make_forecast(model, "2024-01-01T00:00:00", seed)
        target = data_dir / rel
        ds.to_zarr(target, mode="w", consolidated=True, zarr_format=2)
        with xr.open_zarr(target, chunks=None, consolidated=True) as opened:
            store_hashes[(period, model)] = content_sha256(opened)

    # Unpinned persistence store next to pinned stores must NOT be packed.
    pers = _make_forecast("persistence", "2024-01-01T00:00:00", 99)
    pers.to_zarr(
        data_dir / "gate1" / "2024-01_h00-12" / "persistence.zarr",
        mode="w",
        consolidated=True,
        zarr_format=2,
    )

    inventory = {
        "harness_version": {"scores": "2.7.0", "wrfcomp": "0.1.0"},
        "inventory_version": 1,
        "protocol_version": "0.1.0",
        "periods": {},
    }
    for period in ("2024-01", "2024-Q1"):
        inventory["periods"][period] = {
            "truth": "truth/isd_conus_2024.nc",
            "thresholds": "thresholds/tail_p95.nc",
            "other_reference_forecasts": {"0" * 64: f"experiments/persistence_{period}.json"},
            "references": {
                baseline: {
                    "forecast_sha256": store_hashes[(period, baseline)],
                    "model_id": baseline,
                    "period_start": "2024-01-01T00:00:00",
                    "period_end": "2024-01-31T12:00:00",
                    "scorecard": f"experiments/{baseline}_{period}.json",
                    "store": stores[(period, baseline)][0],
                    "thresholds_sha256": thresh_sha,
                    "truth_sha256": truth_sha,
                }
                for baseline in ("climatology", "noaa-gefs")
            },
        }
    ref_path = tmp_path / "REFERENCES.json"
    ref_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {
        "data_dir": data_dir,
        "out_dir": tmp_path / "bundle",
        "ref_path": ref_path,
        "truth_sha": truth_sha,
        "thresh_sha": thresh_sha,
        "store_hashes": store_hashes,
    }


def test_build_and_fetch_roundtrip_reproducible_and_excludes_private_files(synthetic_env, tmp_path):
    out_dir = synthetic_env["out_dir"]
    summary1 = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=out_dir,
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    summary2 = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=out_dir,
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    assert summary1["archive_sha256"] == summary2["archive_sha256"]
    assert summary1["entries"] == 6
    assert summary1["periods"] == ["2024-01", "2024-Q1"]

    archive_path = Path(summary1["archive_path"])
    sha_sidecar = Path(summary1["sha256_sidecar"])
    manifest_sidecar = Path(summary1["manifest_sidecar"])
    assert archive_path.is_file()
    assert sha_sidecar.read_text(encoding="utf-8").startswith(summary1["archive_sha256"])

    with tarfile.open(archive_path, "r:gz") as tf:
        names = tf.getnames()
        assert "MANIFEST.json" in names
        assert "README.md" in names
        assert not any("citizen" in n or ".station_key" in n or "persistence" in n for n in names)
        readme = tf.extractfile("README.md").read().decode("utf-8")
        assert "NOAA" in readme and "CC0-1.0" in readme

    # Fetch with --expected-sha256-file
    dest = tmp_path / "unpacked"
    fetched = fetcher.fetch_scoring_data(
        year=2024,
        data_dir=dest,
        archive=archive_path,
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        expected_sha256_file=sha_sidecar,
    )
    assert fetched["entries_verified"] == 6
    assert len(fetched["installed"]) == 6
    assert not (dest / "citizen").exists()

    # Fetch with --expected-manifest is idempotent when files already match pins.
    again = fetcher.fetch_scoring_data(
        year=2024,
        data_dir=dest,
        archive=archive_path,
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        expected_manifest=manifest_sidecar,
    )
    assert again["installed"] == []

    # Existing modified file is refused without --overwrite, replaced with --overwrite.
    _make_truth(seed=999).to_netcdf(dest / "truth" / "isd_conus_2024.nc", engine="h5netcdf")
    with pytest.raises(fetcher.FetchError, match="different bytes"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=dest,
            archive=archive_path,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=summary1["archive_sha256"],
        )
    replaced = fetcher.fetch_scoring_data(
        year=2024,
        data_dir=dest,
        archive=archive_path,
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        expected_sha256=summary1["archive_sha256"],
        overwrite=True,
    )
    assert "truth/isd_conus_2024.nc" in replaced["installed"]
    with xr.open_dataset(dest / "truth" / "isd_conus_2024.nc") as ds:
        assert content_sha256(ds) == synthetic_env["truth_sha"]


def test_build_refuses_hash_mismatch_sealed_year_and_low_disk(synthetic_env):
    with pytest.raises(builder.BundleError, match="sealed split"):
        builder.build_bundle(
            year=2025,
            data_dir=synthetic_env["data_dir"],
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    with pytest.raises(builder.BundleError, match="No periods pinned"):
        builder.build_bundle(
            year=2023,
            data_dir=synthetic_env["data_dir"],
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    with pytest.raises(builder.BundleError, match="Insufficient free disk space"):
        builder.build_bundle(
            year=2024,
            data_dir=synthetic_env["data_dir"],
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=1e9,
        )
    truth_nc = synthetic_env["data_dir"] / "truth" / "isd_conus_2024.nc"
    _make_truth(seed=777).to_netcdf(truth_nc, engine="h5netcdf")
    with pytest.raises(builder.BundleError, match="Content hash mismatch"):
        builder.build_bundle(
            year=2024,
            data_dir=synthetic_env["data_dir"],
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    assert not (synthetic_env["out_dir"] / "wrfcomp-scoring-2024.tar.gz").exists()


def test_extraction_download_manifest_and_chunk_limits(synthetic_env, tmp_path, monkeypatch):
    """Finding 1: enforce member count, per-member, total expanded, download, manifest & chunk limits."""
    summary = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=synthetic_env["out_dir"],
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    archive_path = Path(summary["archive_path"])
    sha = summary["archive_sha256"]

    # Member count limit
    with pytest.raises(fetcher.FetchError, match="header record count|member count"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "lim_members",
            archive=archive_path,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
            max_members=3,
        )

    # Total expanded bytes limit
    with pytest.raises(fetcher.FetchError, match="expanded"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "lim_total",
            archive=archive_path,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
            max_total_bytes=1024,
        )

    # Per-member size limit
    with pytest.raises(fetcher.FetchError, match="exceeds per-member limit"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "lim_member",
            archive=archive_path,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
            max_member_bytes=512,
        )

    # Bounded MANIFEST.json size limit
    with pytest.raises(fetcher.FetchError, match="MANIFEST.json"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "lim_manifest",
            archive=archive_path,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
            max_manifest_bytes=64,
        )

    # Bounded download size limit (mocked stream, no network)
    archive_bytes = archive_path.read_bytes()

    class _FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        fetcher.urllib.request,
        "urlopen",
        lambda req, timeout=60.0: _FakeResp(archive_bytes),
    )
    with pytest.raises(fetcher.FetchError, match="maximum allowed archive size"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "lim_dl",
            url="https://example.org/wrfcomp-scoring-2024.tar.gz",
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
            max_archive_bytes=1024,
        )

    # Oversized Zarr chunk refused before xarray reads/decompresses
    with pytest.raises(fetcher.FetchError, match="chunk size"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "lim_chunk",
            archive=archive_path,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
            max_chunk_bytes=8,
        )


def test_physical_payload_pinning_strict_zarr_layout_and_trusted_pin(synthetic_env, tmp_path):
    """Finding 2: refuse unrelated files inside Zarr stores, verify references_sha256, require pin."""
    gefs_dir = synthetic_env["data_dir"] / "gate1" / "2024-01_h00-12" / "gefs.zarr"

    # 1. Stray file at Zarr store root refused by builder BEFORE packing
    stray_root = gefs_dir / "unrelated_secret.txt"
    stray_root.write_text("should never be packed", encoding="utf-8")
    with pytest.raises(builder.BundleError, match="unexpected file at Zarr store root"):
        builder.build_bundle(
            year=2024,
            data_dir=synthetic_env["data_dir"],
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    stray_root.unlink()

    # 2. Out-of-bounds chunk / stray file inside Zarr array dir refused by builder
    stray_chunk = gefs_dir / "t2m" / "99.99.99.99"
    stray_chunk.write_bytes(b"smuggled")
    with pytest.raises(builder.BundleError, match="out-of-bounds or unrelated file"):
        builder.build_bundle(
            year=2024,
            data_dir=synthetic_env["data_dir"],
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    stray_chunk.unlink()

    # Build a clean archive now
    summary = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=synthetic_env["out_dir"],
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    good_archive = Path(summary["archive_path"])

    # 3. Fetcher refuses unauthenticated archive by default (no --expected-sha256 / --expected-manifest)
    with pytest.raises(fetcher.FetchError, match="independently trusted physical pin"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "no_pin",
            archive=good_archive,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
        )

    # 4. Fetcher refuses mismatched --expected-manifest
    bad_manifest = tmp_path / "wrong_manifest.json"
    bad_manifest.write_text('{"wrong": true}\n', encoding="utf-8")
    with pytest.raises(fetcher.FetchError, match="does not match trusted --expected-manifest"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "bad_manifest",
            archive=good_archive,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_manifest=bad_manifest,
        )

    # 5. Archive smuggling an unrelated file inside a Zarr store AND adding it to MANIFEST.json is refused
    smuggled_bytes = b"smuggled-payload"
    smuggled_path = "gate1/2024-01_h00-12/gefs.zarr/smuggled.bin"
    smuggled_archive = tmp_path / "smuggled.tar.gz"
    with (
        tarfile.open(good_archive, "r:gz") as src,
        tarfile.open(smuggled_archive, "w:gz", format=tarfile.USTAR_FORMAT) as dst,
    ):
        for member in src.getmembers():
            data = src.extractfile(member).read()
            if member.name == "MANIFEST.json":
                m = json.loads(data.decode("utf-8"))
                for entry in m["entries"]:
                    if entry["path"] == "gate1/2024-01_h00-12/gefs.zarr":
                        entry["files"].append(
                            {
                                "path": smuggled_path,
                                "size_bytes": len(smuggled_bytes),
                                "sha256": hashlib.sha256(smuggled_bytes).hexdigest(),
                            }
                        )
                        entry["size_bytes"] += len(smuggled_bytes)
                data = json.dumps(m).encode("utf-8")
            info = tarfile.TarInfo(member.name)
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))
        extra_info = tarfile.TarInfo(smuggled_path)
        extra_info.size = len(smuggled_bytes)
        dst.addfile(extra_info, io.BytesIO(smuggled_bytes))
    smuggled_sha = hashlib.sha256(smuggled_archive.read_bytes()).hexdigest()
    with pytest.raises(fetcher.FetchError, match="unexpected file at Zarr store root"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "smuggled_dest",
            archive=smuggled_archive,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=smuggled_sha,
        )

    # 6. Fetcher verifies MANIFEST.json references_sha256 against REFERENCES.json
    wrong_ref_archive = tmp_path / "wrong_ref_sha.tar.gz"
    with (
        tarfile.open(good_archive, "r:gz") as src,
        tarfile.open(wrong_ref_archive, "w:gz", format=tarfile.USTAR_FORMAT) as dst,
    ):
        for member in src.getmembers():
            data = src.extractfile(member).read()
            if member.name == "MANIFEST.json":
                m = json.loads(data.decode("utf-8"))
                m["references_sha256"] = "0" * 64
                data = json.dumps(m).encode("utf-8")
            info = tarfile.TarInfo(member.name)
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))
    wrong_ref_sha = hashlib.sha256(wrong_ref_archive.read_bytes()).hexdigest()
    with pytest.raises(fetcher.FetchError, match="references_sha256"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "wrong_ref_dest",
            archive=wrong_ref_archive,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=wrong_ref_sha,
        )


def test_symlink_containment_source_and_destination_and_windows_paths(synthetic_env, tmp_path):
    """Finding 3: refuse symlinked ancestors/files at source & destination, plus Windows path tricks."""
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "escape.nc"
    outside_file.write_bytes(b"outside")

    # 1. Windows path semantics rejected by both builder and fetcher
    bad_paths = ("C:/truth/a.nc", "C:a.nc", "truth\\a.nc", "truth/a:stream.nc", "truth/NUL", "truth/bad ")
    for bad_path in bad_paths:
        with pytest.raises(builder.BundleError):
            builder._validate_relpath(bad_path, "test")
        with pytest.raises(fetcher.FetchError):
            fetcher._validate_relpath(bad_path, "test")

    # 2. Source symlink inside Zarr store refused BEFORE opening with xarray
    gefs_dir = synthetic_env["data_dir"] / "gate1" / "2024-01_h00-12" / "gefs.zarr"
    link_inside = gefs_dir / "t2m" / "999"
    link_inside.symlink_to(outside_file)
    with pytest.raises(builder.BundleError, match="non-regular file refused"):
        builder.build_bundle(
            year=2024,
            data_dir=synthetic_env["data_dir"],
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    link_inside.unlink()

    # 3. Source symlinked ancestor directory refused
    real_truth_dir = synthetic_env["data_dir"] / "truth"
    moved_truth_dir = outside / "moved_truth"
    real_truth_dir.rename(moved_truth_dir)
    real_truth_dir.symlink_to(moved_truth_dir)
    with pytest.raises(builder.BundleError, match="symlink refused"):
        builder.build_bundle(
            year=2024,
            data_dir=synthetic_env["data_dir"],
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    real_truth_dir.unlink()
    moved_truth_dir.rename(real_truth_dir)

    # Build a clean archive for destination symlink tests
    summary = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=synthetic_env["out_dir"],
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    good_archive = Path(summary["archive_path"])
    sha = summary["archive_sha256"]

    # 4. Destination with symlinked ancestor directory (dest/truth -> outside) refused
    dest_link_parent = tmp_path / "dest_link_parent"
    dest_link_parent.mkdir()
    (dest_link_parent / "truth").symlink_to(outside)
    with pytest.raises(fetcher.FetchError, match="symlink refused"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=dest_link_parent,
            archive=good_archive,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
        )
    assert not (outside / "isd_conus_2024.nc").exists()

    # 5. Destination with symlinked target file (dest/truth/isd_conus_2024.nc -> outside_file) refused
    dest_link_file = tmp_path / "dest_link_file"
    (dest_link_file / "truth").mkdir(parents=True)
    (dest_link_file / "truth" / "isd_conus_2024.nc").symlink_to(outside_file)
    with pytest.raises(fetcher.FetchError, match="symlink refused"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=dest_link_file,
            archive=good_archive,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
            overwrite=True,
        )
    assert outside_file.read_bytes() == b"outside"


def test_builder_verifies_exact_bytes_written_into_archive_toctou(synthetic_env, monkeypatch):
    """Finding 4: if a source file changes between hashing and archiving, fail and discard archive."""
    orig_add = builder._add_verified_file_member

    def mutate_before_pack(tf, data_dir, relpath, expected_size, expected_sha256):
        if relpath == "truth/isd_conus_2024.nc":
            p = data_dir / relpath
            raw = bytearray(p.read_bytes())
            # Flip 1 byte keeping identical size to specifically test byte-hash check during streaming.
            raw[-1] ^= 0xFF
            p.write_bytes(bytes(raw))
        return orig_add(tf, data_dir, relpath, expected_size, expected_sha256)

    monkeypatch.setattr(builder, "_add_verified_file_member", mutate_before_pack)

    with pytest.raises(builder.BundleError, match="Source file bytes changed while packing"):
        builder.build_bundle(
            year=2024,
            data_dir=synthetic_env["data_dir"],
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    assert not (synthetic_env["out_dir"] / "wrfcomp-scoring-2024.tar.gz").exists()
    assert not (synthetic_env["out_dir"] / ".wrfcomp-scoring-2024.tar.gz.tmp").exists()


def test_fetch_refuses_tampered_file_spoofed_manifest_and_unsafe_tar(synthetic_env, tmp_path):
    summary = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=synthetic_env["out_dir"],
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    good_archive = Path(summary["archive_path"])

    # 1. Expected archive SHA mismatch
    with pytest.raises(fetcher.FetchError, match="Archive SHA-256 mismatch"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "d1",
            archive=good_archive,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256="0" * 64,
        )

    # 2. Tamper one file inside archive without updating MANIFEST.json
    tampered_file = tmp_path / "tampered_file.tar.gz"
    with (
        tarfile.open(good_archive, "r:gz") as src,
        tarfile.open(tampered_file, "w:gz", format=tarfile.USTAR_FORMAT) as dst,
    ):
        for member in src.getmembers():
            data = src.extractfile(member).read()
            if member.name == "truth/isd_conus_2024.nc":
                data = data[:-4] + b"XXXX"
            info = tarfile.TarInfo(member.name)
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))
    d2 = tmp_path / "d2"
    with pytest.raises(fetcher.FetchError, match="Byte SHA-256 mismatch"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=d2,
            archive=tampered_file,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=hashlib.sha256(tampered_file.read_bytes()).hexdigest(),
        )
    assert not (d2 / "truth").exists()

    # 3. Spoof both truth file AND MANIFEST.json -> caught by REFERENCES.json pin
    alt_truth = tmp_path / "alt_truth.nc"
    _make_truth(seed=555).to_netcdf(alt_truth, engine="h5netcdf")
    alt_bytes = alt_truth.read_bytes()
    with xr.open_dataset(alt_truth) as opened:
        alt_content_sha = content_sha256(opened)
    spoofed = tmp_path / "spoofed_manifest.tar.gz"
    with (
        tarfile.open(good_archive, "r:gz") as src,
        tarfile.open(spoofed, "w:gz", format=tarfile.USTAR_FORMAT) as dst,
    ):
        for member in src.getmembers():
            data = src.extractfile(member).read()
            if member.name == "truth/isd_conus_2024.nc":
                data = alt_bytes
            elif member.name == "MANIFEST.json":
                m = json.loads(data.decode("utf-8"))
                for entry in m["entries"]:
                    if entry["path"] == "truth/isd_conus_2024.nc":
                        entry["content_sha256"] = alt_content_sha
                        entry["size_bytes"] = len(alt_bytes)
                        entry["files"][0]["size_bytes"] = len(alt_bytes)
                        entry["files"][0]["sha256"] = hashlib.sha256(alt_bytes).hexdigest()
                data = json.dumps(m).encode("utf-8")
            info = tarfile.TarInfo(member.name)
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))
    d3 = tmp_path / "d3"
    with pytest.raises(fetcher.FetchError, match="disagrees with REFERENCES.json"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=d3,
            archive=spoofed,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=hashlib.sha256(spoofed.read_bytes()).hexdigest(),
        )
    assert not (d3 / "truth").exists()

    # 4. Path traversal member refused
    traversal = tmp_path / "traversal.tar.gz"
    with (
        gzip.open(traversal, "wb") as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tf,
    ):
        info = tarfile.TarInfo("../escape.txt")
        info.size = 3
        tf.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(fetcher.FetchError, match="unsafe path component"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "d4",
            archive=traversal,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=hashlib.sha256(traversal.read_bytes()).hexdigest(),
        )

    # 5. Extra unmanifested file in archive refused
    extra = tmp_path / "extra.tar.gz"
    with (
        tarfile.open(good_archive, "r:gz") as src,
        tarfile.open(extra, "w:gz", format=tarfile.USTAR_FORMAT) as dst,
    ):
        for member in src.getmembers():
            data = src.extractfile(member).read()
            info = tarfile.TarInfo(member.name)
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))
        sneaky = tarfile.TarInfo("truth/sneaky.txt")
        sneaky.size = 5
        dst.addfile(sneaky, io.BytesIO(b"extra"))
    with pytest.raises(fetcher.FetchError, match="unmanifested extra files"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "d5",
            archive=extra,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=hashlib.sha256(extra.read_bytes()).hexdigest(),
        )


def test_fetch_resolves_zenodo_record_url_with_mock_transport(synthetic_env, tmp_path, monkeypatch):
    summary = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=synthetic_env["out_dir"],
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    archive_bytes = Path(summary["archive_path"]).read_bytes()
    archive_md5 = hashlib.md5(archive_bytes).hexdigest()

    record_api_url = "https://zenodo.org/api/records/12345678"
    file_url = "https://zenodo.org/api/records/12345678/files/wrfcomp-scoring-2024.tar.gz/content"
    record_payload = json.dumps(
        {
            "files": [
                {
                    "key": "wrfcomp-scoring-2024.tar.gz",
                    "checksum": f"md5:{archive_md5}",
                    "links": {"self": file_url},
                },
                {
                    "key": "wrfcomp-scoring-2024.tar.gz.sha256",
                    "links": {"self": file_url + ".sha256"},
                },
            ]
        }
    ).encode("utf-8")

    class _FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(req, timeout=60.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url == record_api_url:
            return _FakeResp(record_payload)
        if url == file_url:
            return _FakeResp(archive_bytes)
        raise AssertionError(f"Unexpected URL requested: {url}")

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", fake_urlopen)

    dest = tmp_path / "from_zenodo"
    result = fetcher.fetch_scoring_data(
        year=2024,
        data_dir=dest,
        url="https://zenodo.org/records/12345678",
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        expected_sha256=summary["archive_sha256"],
    )
    assert result["entries_verified"] == 6
    assert len(result["installed"]) == 6

    with pytest.raises(fetcher.FetchError, match="Refusing URL scheme"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "bad_scheme",
            url="http://zenodo.org/records/12345678",
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=summary["archive_sha256"],
        )


def test_decompression_stream_budget_caps_pax_header_metadata(synthetic_env, tmp_path):
    """Plain USTAR enforcement rejects PAX/GNU headers and enforces stream + metadata budgets."""
    summary = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=synthetic_env["out_dir"],
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    manifest_sidecar = Path(summary["manifest_sidecar"])

    # 0. Verify builder emits pure POSIX USTAR regular-file headers only (no PAX/GNU headers)
    with gzip.open(summary["archive_path"], "rb") as gz:
        raw_tar = gz.read()
    pos = 0
    seen_headers = 0
    while pos + 512 <= len(raw_tar):
        hdr = raw_tar[pos : pos + 512]
        if hdr == b"\x00" * 512:
            break
        assert hdr[257:263] == b"ustar\x00"
        assert hdr[263:265] == b"00"
        assert hdr[156:157] == tarfile.REGTYPE
        size = int(hdr[124:136].rstrip(b"\x00 ").decode("ascii"), 8)
        pos += 512 + ((size + 511) // 512) * 512
        seen_headers += 1
    assert seen_headers >= 8

    # 1. PAX extended header bomb (100 KiB comment) is rejected immediately on the 512-byte header
    # even under the default 10 GiB payload cap.
    bloated = tmp_path / "pax_bomb.tar.gz"
    with (
        gzip.open(bloated, "wb") as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf,
    ):
        info = tarfile.TarInfo("README.md")
        info.size = 2
        info.pax_headers = {"comment": "X" * 100_000}
        tf.addfile(info, io.BytesIO(b"ok"))

    with pytest.raises(fetcher.FetchError, match="Refusing tar PAX/GNU extended header"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "pax_dest",
            archive=bloated,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_manifest=manifest_sidecar,
        )

    # 2. Framing-changing PAX size override header is also refused immediately (no parser split).
    pax_size = tmp_path / "pax_size.tar.gz"
    with (
        gzip.open(pax_size, "wb") as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf,
    ):
        info = tarfile.TarInfo("README.md")
        info.size = 4
        info.pax_headers = {"size": "4"}
        tf.addfile(info, io.BytesIO(b"test"))

    with pytest.raises(fetcher.FetchError, match="Refusing tar PAX/GNU extended header"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "pax_size_dest",
            archive=pax_size,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_manifest=manifest_sidecar,
        )

    # 3. GNU extended header (longname >100 chars -> typeflag 'L') is refused immediately.
    gnu_long = tmp_path / "gnu_long.tar.gz"
    with (
        gzip.open(gnu_long, "wb") as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tf,
    ):
        info = tarfile.TarInfo("truth/" + "a" * 120 + ".nc")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"ok"))

    with pytest.raises(fetcher.FetchError, match="Refusing tar PAX/GNU extended header"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "gnu_dest",
            archive=gnu_long,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_manifest=manifest_sidecar,
        )

    # 4. Non-regular/non-dir USTAR member (symlink) is refused immediately at stream framing.
    ustar_sym = tmp_path / "ustar_sym.tar.gz"
    with (
        gzip.open(ustar_sym, "wb") as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tf,
    ):
        info = tarfile.TarInfo("truth/link.nc")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)

    with pytest.raises(fetcher.FetchError, match="Refusing non-regular tar member"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "sym_dest",
            archive=ustar_sym,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_manifest=manifest_sidecar,
        )

    # 4b. Legacy AREGTYPE (b"\x00") marker (including trailing-slash directory-normalization case
    # with non-zero body size) is refused immediately at the first header check.
    for idx, (arc_name, body) in enumerate([("README.md", b"ok"), ("truth/", b"X" * 512)]):
        areg_tar = tmp_path / f"ustar_areg_{idx}.tar.gz"
        info = tarfile.TarInfo(arc_name)
        info.type = tarfile.AREGTYPE
        info.size = len(body)
        raw_hdr = info.tobuf(tarfile.USTAR_FORMAT)
        assert raw_hdr[156:157] == tarfile.AREGTYPE
        padded_body = body + b"\x00" * ((512 - len(body) % 512) % 512)
        with gzip.open(areg_tar, "wb") as gz:
            gz.write(raw_hdr + padded_body + b"\x00" * 1024)
        with pytest.raises(fetcher.FetchError, match="Refusing non-regular tar member"):
            fetcher.fetch_scoring_data(
                year=2024,
                data_dir=tmp_path / f"areg_dest_{idx}",
                archive=areg_tar,
                references_path=synthetic_env["ref_path"],
                config_path=CONFIG_PATH,
                expected_manifest=manifest_sidecar,
            )

    # 5. Plain USTAR archive exceeding independent tar metadata bytes and header count caps
    # is rejected even under the default 10 GiB payload cap.
    many_ustar = tmp_path / "many_ustar.tar.gz"
    with (
        gzip.open(many_ustar, "wb") as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tf,
    ):
        for idx in range(20):
            info = tarfile.TarInfo(f"truth/part_{idx}.txt")
            info.size = 2
            tf.addfile(info, io.BytesIO(b"ok"))

    with pytest.raises(fetcher.FetchError, match="tar metadata bytes exceeded independent budget"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "many_meta_dest",
            archive=many_ustar,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_manifest=manifest_sidecar,
            max_tar_metadata_bytes=4096,
        )

    with pytest.raises(fetcher.FetchError, match="header record count exceeded independent budget"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=tmp_path / "many_hdr_dest",
            archive=many_ustar,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_manifest=manifest_sidecar,
            max_tar_headers=6,
        )


def test_builder_public_metadata_allowlist_and_nested_netcdf_group(synthetic_env, tmp_path):
    """Re-review item 2: refuse unapproved provenance attrs in NetCDF/Zarr and nested NetCDF groups."""
    data_dir = synthetic_env["data_dir"]
    truth_path = data_dir / "truth" / "isd_conus_2024.nc"
    orig_truth_bytes = truth_path.read_bytes()

    # 1. Unapproved root provenance attribute on NetCDF truth refused
    ds_root_attr = _make_truth(seed=10)
    ds_root_attr.attrs["hostname"] = "private-build-box"
    ds_root_attr.to_netcdf(truth_path, engine="h5netcdf")
    with pytest.raises(builder.BundleError, match="unapproved root NetCDF metadata attributes"):
        builder.build_bundle(
            year=2024,
            data_dir=data_dir,
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )

    # 2. Unapproved variable attribute on NetCDF truth refused
    ds_var_attr = _make_truth(seed=10)
    ds_var_attr["t2m"].attrs["private_note"] = "internal"
    ds_var_attr.to_netcdf(truth_path, engine="h5netcdf")
    with pytest.raises(builder.BundleError, match="unapproved variable NetCDF metadata attributes"):
        builder.build_bundle(
            year=2024,
            data_dir=data_dir,
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )

    # 3. Nested HDF5/NetCDF group hidden inside truth file refused
    truth_path.write_bytes(orig_truth_bytes)
    with h5py.File(truth_path, "a") as h5f:
        grp = h5f.create_group("private_group")
        grp.create_dataset("secret", data=np.array([1, 2, 3], dtype=np.int32))
    with pytest.raises(builder.BundleError, match="nested NetCDF/HDF5 group refused"):
        builder.build_bundle(
            year=2024,
            data_dir=data_dir,
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    truth_path.write_bytes(orig_truth_bytes)

    # 4. Unapproved provenance attribute on Zarr store root (.zattrs) refused
    gefs_path = data_dir / "gate1" / "2024-01_h00-12" / "gefs.zarr"
    orig_zattrs = (gefs_path / ".zattrs").read_bytes()
    orig_zmeta = (gefs_path / ".zmetadata").read_bytes()
    bad_attrs = json.loads(orig_zattrs.decode("utf-8"))
    bad_attrs["local_source_path"] = "/private/cache"
    (gefs_path / ".zattrs").write_text(json.dumps(bad_attrs), encoding="utf-8")
    with pytest.raises(builder.BundleError, match="unapproved metadata attributes refused"):
        builder.build_bundle(
            year=2024,
            data_dir=data_dir,
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    (gefs_path / ".zattrs").write_bytes(orig_zattrs)
    (gefs_path / ".zmetadata").write_bytes(orig_zmeta)

    # 5. Unapproved variable attribute inside Zarr array (.zattrs) refused
    t2m_zattrs_path = gefs_path / "t2m" / ".zattrs"
    orig_t2m_zattrs = t2m_zattrs_path.read_bytes()
    bad_var_attrs = json.loads(orig_t2m_zattrs.decode("utf-8"))
    bad_var_attrs["comment"] = "private note"
    t2m_zattrs_path.write_text(json.dumps(bad_var_attrs), encoding="utf-8")
    with pytest.raises(builder.BundleError, match="unapproved metadata attributes refused"):
        builder.build_bundle(
            year=2024,
            data_dir=data_dir,
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    t2m_zattrs_path.write_bytes(orig_t2m_zattrs)

    # 6. Allowed key with INVALID VALUE: bad t2m units on truth NetCDF (logical hash still matches!)
    ds_bad_units = _make_truth(seed=10)
    ds_bad_units["t2m"].attrs["units"] = "degC"
    ds_bad_units.to_netcdf(truth_path, engine="h5netcdf")
    with pytest.raises(builder.BundleError, match="invalid units"):
        builder.build_bundle(
            year=2024,
            data_dir=data_dir,
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )

    # 7. Allowed key with INVALID VALUE: unapproved free-text provenance in truth 'source' attr
    ds_bad_source = _make_truth(seed=10)
    ds_bad_source.attrs["source"] = "private-host:/tmp/dump"
    ds_bad_source.to_netcdf(truth_path, engine="h5netcdf")
    with pytest.raises(builder.BundleError, match="unapproved truth source"):
        builder.build_bundle(
            year=2024,
            data_dir=data_dir,
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    truth_path.write_bytes(orig_truth_bytes)

    # 8. Allowed key with INVALID VALUE: wrong units on Zarr t2m
    bad_units_zattrs = json.loads(orig_t2m_zattrs.decode("utf-8"))
    bad_units_zattrs["units"] = "fahrenheit"
    t2m_zattrs_path.write_text(json.dumps(bad_units_zattrs), encoding="utf-8")
    with pytest.raises(builder.BundleError, match="invalid units"):
        builder.build_bundle(
            year=2024,
            data_dir=data_dir,
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    t2m_zattrs_path.write_bytes(orig_t2m_zattrs)

    # 9. Allowed key with INVALID VALUE: unapproved model_version free text in Zarr root .zattrs
    bad_ver_attrs = json.loads(orig_zattrs.decode("utf-8"))
    bad_ver_attrs["model_version"] = "private /host/path note"
    (gefs_path / ".zattrs").write_text(json.dumps(bad_ver_attrs), encoding="utf-8")
    with pytest.raises(builder.BundleError, match="unapproved model_version"):
        builder.build_bundle(
            year=2024,
            data_dir=data_dir,
            out_dir=synthetic_env["out_dir"],
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            min_free_gb=0.0,
        )
    (gefs_path / ".zattrs").write_bytes(orig_zattrs)

    # 10. Non-finite (NaN / +inf / -inf) and wrong finite tail_quantile refused on BOTH
    # dataset root and variable attributes, on BOTH build and fetch!
    thresh_path = data_dir / "thresholds" / "tail_p95.nc"
    orig_thresh_bytes = thresh_path.read_bytes()
    good_summary = builder.build_bundle(
        year=2024,
        data_dir=data_dir,
        out_dir=synthetic_env["out_dir"],
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    good_archive = Path(good_summary["archive_path"])

    def _repack_with_threshold_bytes(bad_nc_bytes: bytes, dest_tar: Path) -> str:
        with (
            tarfile.open(good_archive, "r:gz") as src,
            tarfile.open(dest_tar, "w:gz", format=tarfile.USTAR_FORMAT) as dst,
        ):
            for member in src.getmembers():
                data = src.extractfile(member).read()
                if member.name == "thresholds/tail_p95.nc":
                    data = bad_nc_bytes
                elif member.name == "MANIFEST.json":
                    m = json.loads(data.decode("utf-8"))
                    for entry in m["entries"]:
                        if entry["path"] == "thresholds/tail_p95.nc":
                            entry["size_bytes"] = len(bad_nc_bytes)
                            entry["files"][0]["size_bytes"] = len(bad_nc_bytes)
                            entry["files"][0]["sha256"] = hashlib.sha256(bad_nc_bytes).hexdigest()
                    data = json.dumps(m).encode("utf-8")
                info = tarfile.TarInfo(member.name)
                info.size = len(data)
                dst.addfile(info, io.BytesIO(data))
        return hashlib.sha256(dest_tar.read_bytes()).hexdigest()

    for idx, (level, bad_val, expected_msg) in enumerate(
        [
            ("root", float("nan"), "non-finite numeric value"),
            ("var", float("nan"), "non-finite numeric value"),
            ("root", float("inf"), "non-finite numeric value"),
            ("var", float("-inf"), "non-finite numeric value"),
            ("root", 0.90, "tail_quantile"),
            ("var", 0.90, "tail_quantile"),
        ]
    ):
        ds_bad_q = _make_thresholds()
        if level == "root":
            ds_bad_q.attrs["tail_quantile"] = bad_val
        else:
            ds_bad_q["t2m"].attrs["tail_quantile"] = bad_val
        ds_bad_q.to_netcdf(thresh_path, engine="h5netcdf")

        with pytest.raises(builder.BundleError, match=expected_msg):
            builder.build_bundle(
                year=2024,
                data_dir=data_dir,
                out_dir=synthetic_env["out_dir"],
                references_path=synthetic_env["ref_path"],
                config_path=CONFIG_PATH,
                min_free_gb=0.0,
            )

        bad_tar = tmp_path / f"bad_q_{idx}.tar.gz"
        bad_sha = _repack_with_threshold_bytes(thresh_path.read_bytes(), bad_tar)
        with pytest.raises(fetcher.FetchError, match=expected_msg):
            fetcher.fetch_scoring_data(
                year=2024,
                data_dir=tmp_path / f"bad_q_dest_{idx}",
                archive=bad_tar,
                references_path=synthetic_env["ref_path"],
                config_path=CONFIG_PATH,
                expected_sha256=bad_sha,
            )

    thresh_path.write_bytes(orig_thresh_bytes)


def test_existing_destination_physical_byte_hash_and_inventory_verified(synthetic_env, tmp_path):
    """Re-review item 3: existing destination differing in bytes (even if logically equal) is refused."""
    summary = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=synthetic_env["out_dir"],
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        min_free_gb=0.0,
    )
    archive_path = Path(summary["archive_path"])
    sha = summary["archive_sha256"]
    manifest = json.loads(Path(summary["manifest_sidecar"]).read_text(encoding="utf-8"))
    expected_truth_file_sha = next(
        e["files"][0]["sha256"] for e in manifest["entries"] if e["path"] == "truth/isd_conus_2024.nc"
    )

    # Prepopulate destination with logically identical truth written with zlib compression (different bytes).
    dest = tmp_path / "recompressed_dest"
    (dest / "truth").mkdir(parents=True)
    recompressed = dest / "truth" / "isd_conus_2024.nc"
    _make_truth(seed=10).to_netcdf(
        recompressed,
        engine="h5netcdf",
        encoding={"t2m": {"zlib": True, "complevel": 9}},
    )
    with xr.open_dataset(recompressed) as opened:
        assert content_sha256(opened) == synthetic_env["truth_sha"]
    assert hashlib.sha256(recompressed.read_bytes()).hexdigest() != expected_truth_file_sha

    # Without --overwrite: must REFUSE (never report as verified or silently keep differing bytes).
    with pytest.raises(fetcher.FetchError, match="different bytes"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=dest,
            archive=archive_path,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
        )
    assert hashlib.sha256(recompressed.read_bytes()).hexdigest() != expected_truth_file_sha

    # With --overwrite: atomically replaces with exact manifest bytes.
    replaced = fetcher.fetch_scoring_data(
        year=2024,
        data_dir=dest,
        archive=archive_path,
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        expected_sha256=sha,
        overwrite=True,
    )
    assert "truth/isd_conus_2024.nc" in replaced["installed"]
    assert hashlib.sha256(recompressed.read_bytes()).hexdigest() == expected_truth_file_sha

    # Extra stray file planted inside an existing destination Zarr store is also refused without --overwrite.
    stray = dest / "gate1" / "2024-01_h00-12" / "gefs.zarr" / "extra.txt"
    stray.write_text("stray", encoding="utf-8")
    with pytest.raises(fetcher.FetchError, match="different file inventory"):
        fetcher.fetch_scoring_data(
            year=2024,
            data_dir=dest,
            archive=archive_path,
            references_path=synthetic_env["ref_path"],
            config_path=CONFIG_PATH,
            expected_sha256=sha,
        )
    fetcher.fetch_scoring_data(
        year=2024,
        data_dir=dest,
        archive=archive_path,
        references_path=synthetic_env["ref_path"],
        config_path=CONFIG_PATH,
        expected_sha256=sha,
        overwrite=True,
    )
    assert not stray.exists()


@pytest.mark.parametrize("legacy", [False, True], ids=["year-scoped", "v1-single-year"])
def test_install_after_inventory_gains_another_year(synthetic_env, tmp_path, monkeypatch, legacy):
    path = synthetic_env["ref_path"]
    original = path.read_bytes()
    inventory = json.loads(original)
    canonical = json.dumps(inventory["periods"], sort_keys=True, separators=(",", ":")).encode()
    expected_digest = hashlib.sha256(original if legacy else canonical).hexdigest()
    if legacy:
        # Model the old producer's manifest fields; use the normal archive and install paths.
        monkeypatch.setattr(builder, "MANIFEST_VERSION", 1)
        monkeypatch.setattr(builder, "periods_sha256", lambda _: expected_digest)
    bundle = builder.build_bundle(
        year=2024,
        data_dir=synthetic_env["data_dir"],
        out_dir=synthetic_env["out_dir"],
        references_path=path,
        config_path=CONFIG_PATH,
        min_free_gb=0,
    )
    manifest = json.loads(Path(bundle["manifest_sidecar"]).read_bytes())
    assert manifest["references_sha256"] == expected_digest
    assert manifest["manifest_version"] == (1 if legacy else 2)

    for label, entry in list(inventory["periods"].items()):
        inventory["periods"][label.replace("2024", "2023")] = json.loads(
            json.dumps(entry).replace("2024", "2023")
        )
    path.write_text(json.dumps(inventory, indent=4), encoding="utf-8", newline="\n")
    install_args = dict(
        year=2024,
        archive=Path(bundle["archive_path"]),
        expected_sha256=bundle["archive_sha256"],
        references_path=path,
        config_path=CONFIG_PATH,
    )
    installed = fetcher.fetch_scoring_data(data_dir=tmp_path / "valid", **install_args)
    assert installed["entries_verified"] == 6
    assert len(installed["installed"]) == 6

    # Change period metadata that does not alter any of the dataset pins. This
    # must still invalidate the digest, for a monthly or a quarterly reference.
    for label in manifest["periods"]:
        entry = inventory["periods"][label]
        entry["other_reference_forecasts"]["0" * 64] += ".changed"
        path.write_text(json.dumps(inventory), encoding="utf-8", newline="\n")
        with pytest.raises(fetcher.FetchError, match="references_sha256"):
            fetcher.fetch_scoring_data(data_dir=tmp_path / label, **install_args)
        assert not (tmp_path / label / "gate1").exists()
        entry["other_reference_forecasts"]["0" * 64] = entry["other_reference_forecasts"][
            "0" * 64
        ].removesuffix(".changed")
