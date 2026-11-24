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
            "t2m": (("valid_time", "station_id"), rng.normal(280.0, 2.0, (4, 2)).astype(np.float32)),
        },
        coords={
            "valid_time": times,
            "station_id": stations,
            "latitude": ("station_id", np.array([38.85, 33.94], dtype=np.float32)),
            "longitude": ("station_id", np.array([-77.04, -118.41], dtype=np.float32)),
            "elevation_m": ("station_id", np.array([4.0, 38.0], dtype=np.float32)),
        },
    )


def _make_thresholds(seed: int = 2) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    stations = np.array(["72405013743", "72295023174"], dtype="U11")
    return xr.Dataset(
        {
            "t2m": (("station_id", "month"), rng.uniform(290.0, 305.0, (2, 12)).astype(np.float32)),
        },
        coords={"station_id": stations, "month": np.arange(1, 13, dtype=np.int32)},
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
            )
        },
        coords={
            "init_time": inits,
            "lead_time": leads,
            "station_id": stations,
            "ens_member": members,
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
    ref_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    with pytest.raises(fetcher.FetchError, match="different content"):
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
    with pytest.raises(fetcher.FetchError, match="member count"):
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
    with pytest.raises(fetcher.FetchError, match="exceeds limit"):
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
    with tarfile.open(good_archive, "r:gz") as src, tarfile.open(smuggled_archive, "w:gz") as dst:
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
    with tarfile.open(good_archive, "r:gz") as src, tarfile.open(wrong_ref_archive, "w:gz") as dst:
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
    with tarfile.open(good_archive, "r:gz") as src, tarfile.open(tampered_file, "w:gz") as dst:
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
    with tarfile.open(good_archive, "r:gz") as src, tarfile.open(spoofed, "w:gz") as dst:
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
    with gzip.open(traversal, "wb") as gz, tarfile.open(fileobj=gz, mode="w") as tf:
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
    with tarfile.open(good_archive, "r:gz") as src, tarfile.open(extra, "w:gz") as dst:
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
