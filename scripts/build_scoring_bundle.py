"""Build a verifiable, reproducible per-year scoring data bundle for Zenodo archival.

Reads ``experiments/REFERENCES.json``, collects *only* the files and Zarr stores pinned for
``--year`` (truth, p95 tail thresholds, and the pinned monthly/quarterly reference stores),
validates symlink-free source containment and strict Zarr v2 store layouts *before* opening any
dataset, verifies every dataset's harness ``content_sha256`` against the pins, streams files
with ``O_NOFOLLOW`` while verifying the exact bytes written into the archive, verifies the
finished archive by reading it back before atomic publish, and writes one reproducible
``.tar.gz`` archive per year under ``<data-dir>/bundle/`` (plus ``.sha256`` and
``.MANIFEST.json`` sidecars).

Nothing from the sealed split (2025 onward) and no citizen data, candidate stores, or raw caches
are ever read or included.

Usage:
    uv run python -P scripts/build_scoring_bundle.py --year 2024
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.bundle_references import MANIFEST_VERSION, periods_sha256, year_periods
from wrfcomp.harness.align import check_forecast_not_sealed, check_not_sealed
from wrfcomp.harness.config import HarnessConfig, load_config
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.harness.schemas import (
    _UNIT_ALIASES,
    OFFSET_SUFFIX,
    VARIABLE_UNITS,
    SchemaError,
    validate_forecast_dataset,
    validate_thresholds,
    validate_truth_dataset,
)
from wrfcomp.harness.truth import passing_qc_signature

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = ROOT / "experiments" / "REFERENCES.json"
DEFAULT_CONFIG = ROOT / "configs" / "splits.yaml"
DEFAULT_MIN_FREE_GB = 40.0
DEFAULT_COMPRESS_LEVEL = 1
DEFAULT_MAX_CHUNK_BYTES = 256 * 1024**2
DEFAULT_MAX_DATASET_BYTES = 2 * 1024**3
DEFAULT_MAX_METADATA_BYTES = 1 * 1024**2
RECOMMENDED_LICENSE = "CC0-1.0"
REQUIRED_BASELINES = ("climatology", "noaa-gefs")
FORBIDDEN_PATH_PARTS = frozenset(
    {
        "citizen",
        ".station_key",
        "candidates",
        "baselines",
        "raw",
        "parsed",
        "bundle",
    }
)
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)
CHUNK_INDEX_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
ZARR_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
NETCDF_MAGICS = (b"\x89HDF\r\n\x1a\n", b"CDF\x01", b"CDF\x02")
MAX_ATTR_VALUE_LEN = 512

# Approved public metadata allowlists (any unlisted attribute is refused before packing).
ALLOWED_HDF5_ROOT_ATTRS: dict[str, frozenset[str]] = {
    "truth": frozenset(
        {
            "_NCProperties",
            "parser_version",
            "passing_qc_codes",
            "protocol_version",
            "source",
            "truth_tolerance_minutes",
        }
    ),
    "thresholds": frozenset(
        {
            "_NCProperties",
            "protocol_version",
            "tail_quantile",
            "train_end_utc",
        }
    ),
}
_HDF5_STRUCTURAL_VAR_ATTRS = frozenset(
    {
        "CLASS",
        "DIMENSION_LIST",
        "NAME",
        "REFERENCE_LIST",
        "_FillValue",
        "_Netcdf4Coordinates",
        "_Netcdf4Dimid",
        "calendar",
        "coordinates",
    }
)
ALLOWED_HDF5_VAR_ATTRS: dict[str, frozenset[str]] = {
    "truth": _HDF5_STRUCTURAL_VAR_ATTRS | {"long_name", "units"},
    "thresholds": _HDF5_STRUCTURAL_VAR_ATTRS | {"tail_quantile", "train_end_utc", "units"},
}
ALLOWED_ZARR_ROOT_ATTRS = frozenset({"model_id", "model_version"})
ALLOWED_ZARR_VAR_ATTRS = frozenset({"_ARRAY_DIMENSIONS", "calendar", "coordinates", "dtype", "units"})
ALLOWED_ZARR_ZGROUP_KEYS = frozenset({"zarr_format"})
ALLOWED_ZARR_ZARRAY_KEYS = frozenset(
    {
        "chunks",
        "compressor",
        "dimension_separator",
        "dtype",
        "fill_value",
        "filters",
        "order",
        "shape",
        "zarr_format",
    }
)
ALLOWED_ZARR_ZMETADATA_KEYS = frozenset({"metadata", "zarr_consolidated_format"})


class BundleError(RuntimeError):
    """Raised when bundle inputs, pins, disk space, layout, or hashes fail validation."""


APPROVED_TRUTH_SOURCES = frozenset({"NOAA ISD global-hourly (noaa-global-hourly-pds)"})
APPROVED_CALENDARS = frozenset({"proleptic_gregorian", "standard", "gregorian"})
APPROVED_COORD_TOKENS = frozenset({"elevation_m", "latitude", "longitude"})
APPROVED_DIMS = frozenset({"init_time", "lead_time", "station_id", "ens_member", "valid_time", "month"})
APPROVED_LEAD_UNITS = frozenset({"hours", "minutes", "seconds", "nanoseconds"})
NCPROPERTIES_RE = re.compile(r"^version=2,h5netcdf=[0-9.]+,hdf5=[0-9.]+,h5py=[0-9.]+$")
CF_TIME_UNITS_RE = re.compile(
    r"^(?:hours|minutes|seconds|days|nanoseconds) since "
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}$"
)
PARSER_VERSION_RE = re.compile(r"^[0-9]{1,8}$")
MODEL_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
ISO_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?$")


@dataclass(frozen=True)
class PinnedItem:
    relpath: str
    kind: str  # "truth" | "thresholds" | "reference_forecast"
    expected_sha256: str
    pinned_by: tuple[dict[str, str], ...]
    expected_model_id: str | None = None


class _HashingReader:
    """File wrapper that hashes every byte actually read by ``tarfile.addfile``."""

    def __init__(self, raw_fh: io.BufferedReader) -> None:
        self._raw = raw_fh
        self._hasher = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        if chunk:
            self._hasher.update(chunk)
            self.bytes_read += len(chunk)
        return chunk

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def default_data_dir() -> Path:
    env = os.environ.get("WRFCOMP_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "wrfcomp-data"


def _validate_relpath(relpath: str, what: str) -> Path:
    if not isinstance(relpath, str) or not relpath:
        raise BundleError(f"{what}: empty or non-string relative path")
    if "\x00" in relpath or any(ord(ch) < 32 for ch in relpath):
        raise BundleError(f"{what}: control characters are forbidden in path {relpath!r}")
    if relpath.startswith("/") or "\\" in relpath or ":" in relpath or "//" in relpath:
        raise BundleError(f"{what}: unsafe path syntax {relpath!r}")
    win = PureWindowsPath(relpath)
    if win.is_absolute() or win.drive:
        raise BundleError(f"{what}: Windows drive/absolute path forbidden: {relpath!r}")
    raw_parts = relpath.split("/")
    for part in raw_parts:
        if not part or part in (".", ".."):
            raise BundleError(f"{what}: unsafe path component {part!r} in {relpath!r}")
        if part.endswith(" ") or part.endswith("."):
            raise BundleError(f"{what}: Windows-unsafe trailing space/dot in {part!r}")
        stem = part.split(".", 1)[0].lower()
        if stem in WINDOWS_RESERVED_NAMES:
            raise BundleError(f"{what}: Windows reserved device name in {relpath!r}")
    p = Path(*raw_parts)
    lowered = {part.lower() for part in p.parts}
    forbidden = lowered & FORBIDDEN_PATH_PARTS
    if forbidden:
        raise BundleError(f"{what}: forbidden path component {sorted(forbidden)} in {relpath!r}")
    return p


def _open_regular_nofollow(path: Path, what: str) -> tuple[io.BufferedReader, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise BundleError(f"{what}: cannot safely open {path.name}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise BundleError(f"{what}: not a regular file: {path.name}")
        return os.fdopen(fd, "rb"), st
    except Exception:
        os.close(fd)
        raise


def _sha256_file(path: Path) -> str:
    fh, _ = _open_regular_nofollow(path, f"file {path.name}")
    h = hashlib.sha256()
    with fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_bounded_json(path: Path, max_bytes: int, what: str) -> Any:
    fh, st = _open_regular_nofollow(path, what)
    with fh:
        if st.st_size > max_bytes:
            raise BundleError(f"{what}: metadata file exceeds {max_bytes} bytes")
        raw = fh.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise BundleError(f"{what}: metadata file exceeds {max_bytes} bytes")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError(f"{what}: invalid JSON: {exc}") from exc


def _assert_source_contained(root: Path, relpath: str, what: str) -> Path:
    """Verify every ancestor and target under ``root`` exists without following any symlink."""
    rel = _validate_relpath(relpath, what)
    root_real = root.resolve(strict=True)
    cur = root
    for idx, part in enumerate(rel.parts):
        cur = cur / part
        if not os.path.lexists(cur):
            raise BundleError(f"{what}: missing path {relpath}")
        st = os.lstat(cur)
        if stat.S_ISLNK(st.st_mode):
            raise BundleError(f"{what}: symlink refused at {cur.relative_to(root).as_posix()}")
        is_last = idx == len(rel.parts) - 1
        if not is_last and not stat.S_ISDIR(st.st_mode):
            raise BundleError(f"{what}: non-directory ancestor in {relpath}")
        if not cur.resolve().is_relative_to(root_real):
            raise BundleError(f"{what}: path escapes data root: {relpath}")
    return cur


def _as_text(val: Any, what: str) -> str:
    if isinstance(val, bytes | np.bytes_):
        try:
            text = bytes(val).decode("ascii")
        except UnicodeDecodeError as exc:
            raise BundleError(f"{what}: non-ASCII bytes value refused") from exc
    elif isinstance(val, str | np.str_):
        text = str(val)
    else:
        raise BundleError(f"{what}: expected string value, got {type(val).__name__}")
    if len(text) > MAX_ATTR_VALUE_LEN or "\x00" in text:
        raise BundleError(f"{what}: string too large or contains NUL")
    return text


def _as_scalar_float(val: Any, what: str) -> float:
    arr = np.asarray(val)
    if arr.size != 1 or arr.dtype.kind not in "iuf":
        raise BundleError(f"{what}: expected numeric scalar")
    out = float(arr.reshape(-1)[0])
    if not math.isfinite(out):
        raise BundleError(f"{what}: non-finite numeric value {out!r} refused")
    return out


def _as_scalar_int(val: Any, what: str) -> int:
    arr = np.asarray(val)
    if arr.size != 1 or arr.dtype.kind not in "iu":
        raise BundleError(f"{what}: expected integer scalar")
    return int(arr.reshape(-1)[0])


def _validate_train_end_value(val: Any, cfg: HarnessConfig, what: str) -> None:
    text = _as_text(val, what)
    if not ISO_TIMESTAMP_RE.fullmatch(text):
        raise BundleError(f"{what}: invalid ISO timestamp {text!r}")
    assert cfg.splits.train.end is not None
    if pd.Timestamp(text) != pd.Timestamp(cfg.splits.train.end):
        raise BundleError(
            f"{what}: train_end_utc {text!r} does not match config {pd.Timestamp(cfg.splits.train.end)}"
        )


def _validate_var_attr_value(
    var_name: str,
    attr_name: str,
    val: Any,
    *,
    kind: str,
    cfg: HarnessConfig,
    what: str,
) -> None:
    if attr_name == "units":
        text = _as_text(val, what)
        if var_name in VARIABLE_UNITS:
            allowed = _UNIT_ALIASES[VARIABLE_UNITS[var_name]]
            if text not in allowed:
                raise BundleError(
                    f"{what}: invalid units {text!r} for {var_name} (expected one of {sorted(allowed)})"
                )
        elif var_name.endswith(OFFSET_SUFFIX) and var_name[: -len(OFFSET_SUFFIX)] in VARIABLE_UNITS:
            if text != "minutes":
                raise BundleError(f"{what}: invalid units {text!r} for offset {var_name}")
        elif var_name == "elevation_m":
            if text != "m":
                raise BundleError(f"{what}: invalid units {text!r} for elevation_m")
        elif var_name == "latitude":
            if text != "degrees_north":
                raise BundleError(f"{what}: invalid units {text!r} for latitude")
        elif var_name == "longitude":
            if text != "degrees_east":
                raise BundleError(f"{what}: invalid units {text!r} for longitude")
        elif var_name in ("valid_time", "init_time"):
            if not CF_TIME_UNITS_RE.fullmatch(text):
                raise BundleError(f"{what}: invalid CF time units {text!r}")
        elif var_name == "lead_time":
            if text not in APPROVED_LEAD_UNITS:
                raise BundleError(f"{what}: invalid lead_time units {text!r}")
        else:
            raise BundleError(f"{what}: unexpected units attribute on {var_name!r}")
        return

    if attr_name == "long_name":
        if kind != "truth" or not var_name.endswith(OFFSET_SUFFIX):
            raise BundleError(f"{what}: long_name is only approved on truth offset variables")
        base = var_name[: -len(OFFSET_SUFFIX)]
        expected = f"Observation time offset from valid_time for {base} (-32768 when missing)"
        text = _as_text(val, what)
        if text != expected:
            raise BundleError(f"{what}: unapproved long_name {text!r}")
        return

    if attr_name == "calendar":
        if var_name not in ("valid_time", "init_time"):
            raise BundleError(f"{what}: calendar only approved on time coordinates")
        text = _as_text(val, what)
        if text not in APPROVED_CALENDARS:
            raise BundleError(f"{what}: unapproved calendar {text!r}")
        return

    if attr_name == "coordinates":
        text = _as_text(val, what)
        tokens = text.split()
        if not tokens or not set(tokens).issubset(APPROVED_COORD_TOKENS):
            raise BundleError(f"{what}: unapproved coordinates attribute {text!r}")
        return

    if attr_name == "tail_quantile":
        if kind != "thresholds":
            raise BundleError(f"{what}: tail_quantile only approved on thresholds")
        q = _as_scalar_float(val, what)
        expected_q = float(cfg.gate.tail_quantile)
        if not (math.isfinite(q) and math.isfinite(expected_q) and q == expected_q):
            raise BundleError(f"{what}: tail_quantile {q!r} != {expected_q!r}")
        return

    if attr_name == "train_end_utc":
        if kind != "thresholds":
            raise BundleError(f"{what}: train_end_utc only approved on thresholds")
        _validate_train_end_value(val, cfg, what)
        return

    if attr_name == "_ARRAY_DIMENSIONS":
        ok_dims = isinstance(val, list) and bool(val) and all(
            isinstance(d, str) and d in APPROVED_DIMS for d in val
        )
        if not ok_dims:
            raise BundleError(f"{what}: unapproved _ARRAY_DIMENSIONS {val!r}")
        return

    if attr_name == "dtype":
        text = _as_text(val, what)
        if var_name != "lead_time" or text != "timedelta64[ns]":
            raise BundleError(f"{what}: unapproved dtype attribute {text!r} on {var_name!r}")
        return

    if attr_name == "CLASS":
        text = _as_text(val, what)
        if text != "DIMENSION_SCALE":
            raise BundleError(f"{what}: unapproved HDF5 CLASS {text!r}")
        return

    if attr_name == "NAME":
        text = _as_text(val, what)
        if text != var_name and not text.startswith("This is a netCDF dimension"):
            raise BundleError(f"{what}: unapproved HDF5 NAME {text!r}")
        return

    if attr_name in (
        "_Netcdf4Dimid",
        "_Netcdf4Coordinates",
        "_FillValue",
        "DIMENSION_LIST",
        "REFERENCE_LIST",
    ):
        return

    raise BundleError(f"{what}: unapproved attribute {attr_name!r}")


def _validate_netcdf_root_attr_value(
    attr_name: str, val: Any, *, kind: str, cfg: HarnessConfig, what: str
) -> None:
    if attr_name == "_NCProperties":
        text = _as_text(val, what)
        if not NCPROPERTIES_RE.fullmatch(text):
            raise BundleError(f"{what}: unapproved _NCProperties value {text!r}")
        return
    if attr_name == "protocol_version":
        text = _as_text(val, what)
        if text != cfg.protocol_version:
            raise BundleError(f"{what}: protocol_version {text!r} != {cfg.protocol_version!r}")
        return
    if kind == "truth":
        if attr_name == "source":
            text = _as_text(val, what)
            if text not in APPROVED_TRUTH_SOURCES:
                raise BundleError(f"{what}: unapproved truth source {text!r}")
            return
        if attr_name == "parser_version":
            text = _as_text(val, what)
            if not PARSER_VERSION_RE.fullmatch(text):
                raise BundleError(f"{what}: invalid parser_version {text!r}")
            return
        if attr_name == "truth_tolerance_minutes":
            tol = _as_scalar_int(val, what)
            if tol != int(cfg.evaluation.truth_tolerance_minutes):
                raise BundleError(
                    f"{what}: truth_tolerance_minutes {tol} != {cfg.evaluation.truth_tolerance_minutes}"
                )
            return
        if attr_name == "passing_qc_codes":
            text = _as_text(val, what)
            if text != passing_qc_signature():
                raise BundleError(f"{what}: unapproved passing_qc_codes {text!r}")
            return
    if kind == "thresholds":
        if attr_name == "tail_quantile":
            q = _as_scalar_float(val, what)
            expected_q = float(cfg.gate.tail_quantile)
            if not (math.isfinite(q) and math.isfinite(expected_q) and q == expected_q):
                raise BundleError(f"{what}: tail_quantile {q!r} != {expected_q!r}")
            return
        if attr_name == "train_end_utc":
            _validate_train_end_value(val, cfg, what)
            return
    raise BundleError(f"{what}: unapproved root attribute {attr_name!r}")


def _validate_zarr_root_attrs(
    attrs: Any,
    *,
    expected_model_id: str | None,
    what: str,
) -> dict[str, Any]:
    if not isinstance(attrs, dict):
        raise BundleError(f"{what}: must be a JSON object")
    disallowed = sorted(set(attrs) - ALLOWED_ZARR_ROOT_ATTRS)
    if disallowed:
        raise BundleError(f"{what}: unapproved metadata attributes refused: {disallowed}")
    model_id = _as_text(attrs.get("model_id"), f"{what}['model_id']")
    if expected_model_id is not None and model_id != expected_model_id:
        raise BundleError(f"{what}: model_id {model_id!r} != pinned {expected_model_id!r}")
    if model_id not in REQUIRED_BASELINES:
        raise BundleError(f"{what}: unapproved reference model_id {model_id!r}")
    model_ver = _as_text(attrs.get("model_version"), f"{what}['model_version']")
    if not MODEL_VERSION_RE.fullmatch(model_ver):
        raise BundleError(f"{what}: unapproved model_version {model_ver!r}")
    return attrs


def _validate_zarr_var_attrs(
    attrs: Any,
    *,
    var_name: str,
    cfg: HarnessConfig,
    what: str,
) -> dict[str, Any]:
    if not isinstance(attrs, dict):
        raise BundleError(f"{what}: must be a JSON object")
    disallowed = sorted(set(attrs) - ALLOWED_ZARR_VAR_ATTRS)
    if disallowed:
        raise BundleError(f"{what}: unapproved metadata attributes refused: {disallowed}")
    for key, val in attrs.items():
        _validate_var_attr_value(
            var_name, key, val, kind="reference_forecast", cfg=cfg, what=f"{what}[{key!r}]"
        )
    return attrs


def _validate_netcdf_file(
    data_dir: Path,
    relpath: str,
    *,
    kind: str,
    cfg: HarnessConfig,
) -> tuple[str, int, list[dict[str, Any]], set[str]]:
    target = _assert_source_contained(data_dir, relpath, f"netcdf {relpath}")
    st = os.lstat(target)
    if not stat.S_ISREG(st.st_mode):
        raise BundleError(f"{relpath}: expected a regular NetCDF file")
    fh, fst = _open_regular_nofollow(target, f"netcdf {relpath}")
    h = hashlib.sha256()
    total = 0
    with fh:
        head = fh.read(8)
        if head != NETCDF_MAGICS[0]:
            raise BundleError(f"{relpath}: expected an HDF5/NetCDF4 file")
        h.update(head)
        total += len(head)
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    if total != fst.st_size:
        raise BundleError(f"{relpath}: file size changed while hashing")

    allowed_root = ALLOWED_HDF5_ROOT_ATTRS.get(kind)
    allowed_var = ALLOWED_HDF5_VAR_ATTRS.get(kind)
    if allowed_root is None or allowed_var is None:
        raise BundleError(f"{relpath}: unsupported NetCDF dataset kind {kind!r}")

    try:
        with h5py.File(target, "r") as h5f:
            disallowed_root = sorted(set(h5f.attrs.keys()) - allowed_root)
            if disallowed_root:
                raise BundleError(
                    f"{relpath}: unapproved root NetCDF metadata attributes refused: {disallowed_root}"
                )
            for attr_name in h5f.attrs.keys():
                _validate_netcdf_root_attr_value(
                    attr_name,
                    h5f.attrs[attr_name],
                    kind=kind,
                    cfg=cfg,
                    what=f"{relpath} root attr {attr_name!r}",
                )

            hdf5_datasets: set[str] = set()
            for name in sorted(h5f.keys()):
                link = h5f.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise BundleError(f"{relpath}: non-hardlink HDF5 entry refused: {name!r}")
                obj = h5f[name]
                if isinstance(obj, h5py.Group):
                    raise BundleError(f"{relpath}: nested NetCDF/HDF5 group refused: {name!r}")
                if not isinstance(obj, h5py.Dataset):
                    raise BundleError(f"{relpath}: unsupported HDF5 object refused: {name!r}")
                disallowed_var = sorted(set(obj.attrs.keys()) - allowed_var)
                if disallowed_var:
                    raise BundleError(
                        f"{relpath}/{name}: unapproved variable NetCDF metadata attributes refused: "
                        f"{disallowed_var}"
                    )
                for attr_name in obj.attrs.keys():
                    _validate_var_attr_value(
                        name,
                        attr_name,
                        obj.attrs[attr_name],
                        kind=kind,
                        cfg=cfg,
                        what=f"{relpath}/{name} attr {attr_name!r}",
                    )
                hdf5_datasets.add(name)
    except BundleError:
        raise
    except Exception as exc:
        raise BundleError(f"{relpath}: failed to inspect HDF5/NetCDF4 structure: {exc}") from exc

    return (
        "netcdf",
        total,
        [{"path": relpath, "size_bytes": total, "sha256": h.hexdigest()}],
        hdf5_datasets,
    )


def _validate_zarr_store(
    data_dir: Path,
    relpath: str,
    *,
    cfg: HarnessConfig,
    expected_model_id: str | None = None,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
    max_metadata_bytes: int = DEFAULT_MAX_METADATA_BYTES,
) -> tuple[str, int, list[dict[str, Any]], set[str]]:
    """Strictly validate a Zarr v2 directory tree BEFORE opening it with xarray.

    Refuses any symlink, non-regular file, unexpected file/directory, out-of-bounds chunk key,
    inconsistent ``.zmetadata``, or oversized chunk/array. Returns
    ``("zarr", total_bytes, files_manifest, array_names)``.
    """
    store = _assert_source_contained(data_dir, relpath, f"zarr store {relpath}")
    root_real = data_dir.resolve(strict=True)
    if not stat.S_ISDIR(os.lstat(store).st_mode):
        raise BundleError(f"{relpath}: expected a Zarr store directory")

    entries = sorted(store.iterdir(), key=lambda p: p.name)
    root_files: dict[str, Path] = {}
    var_dirs: dict[str, Path] = {}
    for child in entries:
        st = os.lstat(child)
        if stat.S_ISLNK(st.st_mode):
            raise BundleError(f"{relpath}: symlink refused inside store: {child.name}")
        if not child.resolve().is_relative_to(root_real):
            raise BundleError(f"{relpath}: store entry escapes data root: {child.name}")
        if stat.S_ISREG(st.st_mode):
            if child.name not in (".zgroup", ".zattrs", ".zmetadata"):
                raise BundleError(f"{relpath}: unexpected file at Zarr store root: {child.name!r}")
            root_files[child.name] = child
        elif stat.S_ISDIR(st.st_mode):
            if not ZARR_VAR_NAME_RE.fullmatch(child.name):
                raise BundleError(f"{relpath}: unexpected subdirectory in Zarr store: {child.name!r}")
            var_dirs[child.name] = child
        else:
            raise BundleError(f"{relpath}: non-regular entry in Zarr store: {child.name!r}")

    if ".zgroup" not in root_files:
        raise BundleError(f"{relpath}: missing .zgroup in Zarr v2 store")
    zgroup = _read_bounded_json(root_files[".zgroup"], max_metadata_bytes, f"{relpath}/.zgroup")
    if (
        not isinstance(zgroup, dict)
        or set(zgroup) != ALLOWED_ZARR_ZGROUP_KEYS
        or zgroup.get("zarr_format") != 2
    ):
        raise BundleError(f"{relpath}: only Zarr v2 groups with standard keys are supported")

    parsed_metadata: dict[str, Any] = {".zgroup": zgroup}
    if ".zattrs" in root_files:
        zattrs = _validate_zarr_root_attrs(
            _read_bounded_json(root_files[".zattrs"], max_metadata_bytes, f"{relpath}/.zattrs"),
            expected_model_id=expected_model_id,
            what=f"{relpath}/.zattrs",
        )
        parsed_metadata[".zattrs"] = zattrs
    else:
        raise BundleError(f"{relpath}: missing .zattrs with model_id/model_version")

    if not var_dirs:
        raise BundleError(f"{relpath}: Zarr store contains no arrays")

    allowed_files: list[Path] = [root_files[k] for k in sorted(root_files)]
    total_decoded_bytes = 0

    for var_name, vdir in sorted(var_dirs.items()):
        zarray_path = vdir / ".zarray"
        if not os.path.lexists(zarray_path) or not stat.S_ISREG(os.lstat(zarray_path).st_mode):
            raise BundleError(f"{relpath}/{var_name}: missing regular .zarray metadata")
        za = _read_bounded_json(zarray_path, max_metadata_bytes, f"{relpath}/{var_name}/.zarray")
        if (
            not isinstance(za, dict)
            or not set(za).issubset(ALLOWED_ZARR_ZARRAY_KEYS)
            or za.get("zarr_format") != 2
        ):
            raise BundleError(f"{relpath}/{var_name}: unsupported .zarray format or keys")
        shape = za.get("shape")
        chunks = za.get("chunks")
        dtype_raw = za.get("dtype")
        sep = za.get("dimension_separator", ".")
        if sep not in (".", "/"):
            raise BundleError(f"{relpath}/{var_name}: unsupported dimension_separator {sep!r}")
        if (
            not isinstance(shape, list)
            or not isinstance(chunks, list)
            or len(shape) != len(chunks)
            or any(not isinstance(s, int) or s < 0 for s in shape)
            or any(not isinstance(c, int) or c <= 0 for c in chunks)
        ):
            raise BundleError(f"{relpath}/{var_name}: invalid shape/chunks in .zarray")
        try:
            itemsize = max(int(np.dtype(dtype_raw).itemsize), 8)
        except Exception as exc:
            raise BundleError(f"{relpath}/{var_name}: invalid dtype {dtype_raw!r}") from exc
        chunk_elems = math.prod(chunks) if chunks else 1
        array_elems = math.prod(shape) if shape else 1
        if chunk_elems * itemsize > max_chunk_bytes:
            raise BundleError(
                f"{relpath}/{var_name}: chunk size {chunk_elems * itemsize} exceeds limit {max_chunk_bytes}"
            )
        total_decoded_bytes += array_elems * itemsize
        if total_decoded_bytes > max_dataset_bytes:
            raise BundleError(
                f"{relpath}: decoded dataset size exceeds limit {max_dataset_bytes} bytes"
            )

        parsed_metadata[f"{var_name}/.zarray"] = za
        grid_limits = [math.ceil(s / c) for s, c in zip(shape, chunks, strict=True)] if shape else [1]

        for dirpath, dirnames, filenames in os.walk(vdir, followlinks=False):
            dirnames.sort()
            filenames.sort()
            cur_dir = Path(dirpath)
            if cur_dir != vdir and sep != "/":
                raise BundleError(
                    f"{relpath}/{var_name}: unexpected subdirectory {cur_dir.relative_to(vdir).as_posix()!r}"
                )
            for dname in dirnames:
                dpath = cur_dir / dname
                if stat.S_ISLNK(os.lstat(dpath).st_mode):
                    raise BundleError(f"{relpath}/{var_name}: symlink directory refused: {dname!r}")
                if sep != "/":
                    raise BundleError(f"{relpath}/{var_name}: unexpected nested directory {dname!r}")
                if not CHUNK_INDEX_RE.fullmatch(dname):
                    raise BundleError(f"{relpath}/{var_name}: invalid chunk directory name {dname!r}")
            for fname in filenames:
                fpath = cur_dir / fname
                fst = os.lstat(fpath)
                if stat.S_ISLNK(fst.st_mode) or not stat.S_ISREG(fst.st_mode):
                    raise BundleError(f"{relpath}/{var_name}: non-regular file refused: {fname!r}")
                if not fpath.resolve().is_relative_to(root_real):
                    raise BundleError(f"{relpath}/{var_name}: file escapes data root: {fname!r}")
                inner_rel = fpath.relative_to(vdir).as_posix()
                if inner_rel == ".zarray":
                    allowed_files.append(fpath)
                    continue
                if inner_rel == ".zattrs":
                    vattrs = _validate_zarr_var_attrs(
                        _read_bounded_json(fpath, max_metadata_bytes, f"{relpath}/{var_name}/.zattrs"),
                        var_name=var_name,
                        cfg=cfg,
                        what=f"{relpath}/{var_name}/.zattrs",
                    )
                    parsed_metadata[f"{var_name}/.zattrs"] = vattrs
                    allowed_files.append(fpath)
                    continue
                parts = inner_rel.split(sep)
                if len(parts) != len(grid_limits):
                    raise BundleError(
                        f"{relpath}/{var_name}: unrelated file or malformed chunk key {inner_rel!r}"
                    )
                for part, lim in zip(parts, grid_limits, strict=True):
                    if not CHUNK_INDEX_RE.fullmatch(part) or int(part) >= lim:
                        raise BundleError(
                            f"{relpath}/{var_name}: out-of-bounds or unrelated file {inner_rel!r}"
                        )
                allowed_files.append(fpath)

    if ".zmetadata" in root_files:
        zmeta = _read_bounded_json(root_files[".zmetadata"], max_metadata_bytes, f"{relpath}/.zmetadata")
        if (
            not isinstance(zmeta, dict)
            or set(zmeta) != ALLOWED_ZARR_ZMETADATA_KEYS
            or zmeta.get("zarr_consolidated_format") != 1
            or not isinstance(zmeta.get("metadata"), dict)
        ):
            raise BundleError(f"{relpath}/.zmetadata is malformed")
        meta_map = zmeta["metadata"]
        if set(meta_map) != set(parsed_metadata):
            raise BundleError(
                f"{relpath}/.zmetadata keys do not match on-disk Zarr metadata files"
            )
        for mkey, mval in meta_map.items():
            if mval != parsed_metadata[mkey]:
                raise BundleError(f"{relpath}/.zmetadata disagrees with on-disk {mkey}")

    files_manifest: list[dict[str, Any]] = []
    total_size = 0
    for fpath in sorted(allowed_files, key=lambda p: p.relative_to(data_dir).as_posix()):
        rel = fpath.relative_to(data_dir).as_posix()
        _validate_relpath(rel, f"store member {rel}")
        fh, fst = _open_regular_nofollow(fpath, f"store member {rel}")
        h = hashlib.sha256()
        read_len = 0
        with fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                read_len += len(chunk)
        if read_len != fst.st_size:
            raise BundleError(f"{rel}: file size changed while hashing")
        total_size += read_len
        files_manifest.append({"path": rel, "size_bytes": read_len, "sha256": h.hexdigest()})

    return "zarr", total_size, files_manifest, set(var_dirs)


def _open_dataset(path: Path, fmt: str, what: str) -> xr.Dataset:
    try:
        if fmt == "zarr":
            try:
                return xr.open_zarr(path, chunks=None, consolidated=True)
            except (KeyError, ValueError, FileNotFoundError):
                return xr.open_zarr(path, chunks=None, consolidated=False)
        return xr.open_dataset(path)
    except Exception as exc:
        raise BundleError(f"{what}: failed to open {path.name}: {exc}") from exc


def _check_dataset_bounds(
    ds: xr.Dataset,
    relpath: str,
    *,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
) -> None:
    total = 0
    for vname, var in ds.variables.items():
        if any(s < 0 for s in var.shape):
            raise BundleError(f"{relpath}: negative dimension on {vname}")
        itemsize = max(int(var.dtype.itemsize), 8)
        total += int(var.size) * itemsize
        if total > max_dataset_bytes:
            raise BundleError(
                f"{relpath}: decoded dataset size exceeds limit {max_dataset_bytes} bytes"
            )


def _check_not_sealed(ds: xr.Dataset, kind: str, relpath: str, cfg: HarnessConfig) -> None:
    sealed_start = cfg.splits.sealed_test.start
    if kind == "truth":
        if "valid_time" not in ds.coords:
            raise BundleError(f"{relpath}: truth dataset missing valid_time coordinate")
        check_not_sealed(ds["valid_time"].values, what=f"truth {relpath}", sealed_start=sealed_start)
    elif kind == "reference_forecast":
        check_forecast_not_sealed(ds, what=f"reference {relpath}", sealed_start=sealed_start)


def collect_pinned_items(
    references_path: Path,
    year: int,
    cfg: HarnessConfig,
) -> tuple[dict[str, Any], list[str], list[PinnedItem]]:
    """Read ``REFERENCES.json`` and return ``(inventory, periods, pinned_items)`` for ``year``."""
    sealed_dt = cfg.splits.sealed_test.start_utc
    sealed_year = sealed_dt.year
    sealed_iso = sealed_dt.replace(tzinfo=None).isoformat()
    if year >= sealed_year:
        raise BundleError(f"Refusing year {year}: sealed split starts at {sealed_iso}")
    try:
        inventory = json.loads(references_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"Cannot read references inventory: {exc}") from exc
    if not isinstance(inventory, dict) or not isinstance(inventory.get("periods"), dict):
        raise BundleError("Invalid REFERENCES.json structure")

    selected = year_periods(inventory, year)
    seen: dict[str, tuple[str, str, list[dict[str, str]], str | None]] = {}

    def record(
        relpath: str, kind: str, digest: str, citation: dict[str, str], model_id: str | None = None
    ) -> None:
        _validate_relpath(relpath, f"{kind} ({citation['period']})")
        if not isinstance(digest, str) or len(digest) != 64:
            raise BundleError(f"Invalid sha256 pin for {relpath}: {digest!r}")
        if relpath in seen:
            prev_kind, prev_digest, citations, prev_model = seen[relpath]
            if prev_kind != kind:
                raise BundleError(f"Conflicting kinds for {relpath}: {prev_kind} vs {kind}")
            if prev_digest != digest:
                raise BundleError(f"Conflicting sha256 pins for {relpath}: {prev_digest} vs {digest}")
            if prev_model != model_id:
                raise BundleError(f"Conflicting model_id pins for {relpath}: {prev_model} vs {model_id}")
            citations.append(citation)
        else:
            seen[relpath] = (kind, digest, [citation], model_id)

    for period, entry in selected.items():
        refs = entry.get("references")
        if not isinstance(refs, dict) or set(refs) != set(REQUIRED_BASELINES):
            raise BundleError(f"Period {period} must pin exactly {', '.join(REQUIRED_BASELINES)}")
        truth_rel = entry["truth"]
        thresh_rel = entry["thresholds"]
        for baseline in REQUIRED_BASELINES:
            rec = refs[baseline]
            for time_key in ("period_start", "period_end"):
                stamp = str(rec.get(time_key, ""))
                if stamp >= sealed_iso:
                    raise BundleError(f"Period {period} {baseline} {time_key} reaches sealed split: {stamp}")
            scorecard = str(rec["scorecard"])
            record(
                truth_rel,
                "truth",
                rec["truth_sha256"],
                {
                    "period": period,
                    "baseline": baseline,
                    "field": "truth_sha256",
                    "scorecard": scorecard,
                },
            )
            record(
                thresh_rel,
                "thresholds",
                rec["thresholds_sha256"],
                {
                    "period": period,
                    "baseline": baseline,
                    "field": "thresholds_sha256",
                    "scorecard": scorecard,
                },
            )
            record(
                rec["store"],
                "reference_forecast",
                rec["forecast_sha256"],
                {
                    "period": period,
                    "baseline": baseline,
                    "field": "forecast_sha256",
                    "scorecard": scorecard,
                },
                model_id=str(rec["model_id"]),
            )

    if not selected:
        raise BundleError(f"No periods pinned for year {year} in {references_path.name}")

    items = [
        PinnedItem(
            relpath=relpath,
            kind=kind,
            expected_sha256=digest,
            pinned_by=tuple(sorted(citations, key=lambda c: (c["period"], c["baseline"], c["field"]))),
            expected_model_id=model_id,
        )
        for relpath, (kind, digest, citations, model_id) in sorted(seen.items())
    ]
    return inventory, list(selected), items


def _check_free_space(path: Path, min_free_gb: float, required_extra_bytes: int = 0) -> float:
    """Ensure free space on the filesystem holding ``path`` stays above ``min_free_gb``."""
    anchor = path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise BundleError(
            f"Insufficient free disk space: {free_gb:.2f} GiB available, "
            f"minimum required is {min_free_gb:.2f} GiB"
        )
    projected_gb = (usage.free - required_extra_bytes) / (1024**3)
    if required_extra_bytes > 0 and projected_gb < min_free_gb:
        raise BundleError(
            f"Projected free disk space after packing ({projected_gb:.2f} GiB) would fall below "
            f"the {min_free_gb:.2f} GiB floor"
        )
    return free_gb


def render_bundle_readme(
    *,
    year: int,
    periods: list[str],
    entries: list[dict[str, Any]],
    harness_version: dict[str, Any],
    protocol_version: str,
) -> str:
    total_bytes = sum(int(e["size_bytes"]) for e in entries)
    total_mib = total_bytes / (1024**2)
    return (
        f"# WRFComp {year} Scoring Data Bundle\n\n"
        f"This archive contains the exact verification datasets pinned by `experiments/REFERENCES.json`\n"
        f"for scoring candidate weather forecasts on **{year}** under WRFComp harness version\n"
        f"`{harness_version.get('wrfcomp', 'unknown')}` (protocol `{protocol_version}`).\n\n"
        f"## Contents\n\n"
        f"- **Periods covered ({len(periods)}):** {', '.join(periods)}\n"
        f"- **Pinned datasets:** {len(entries)} ({total_mib:.1f} MiB uncompressed)\n"
        f"  - Station ground truth (`truth/`): hourly NOAA ISD station observations for {year}\n"
        f"  - Tail thresholds (`thresholds/`): monthly station-level 95th-percentile thresholds\n"
        f"  - Reference forecasts (`gate1/`): pinned `climatology.zarr` and `gefs.zarr` stores\n"
        f"- **`MANIFEST.json`:** per-dataset harness `content_sha256`, per-file byte SHA-256, sizes, and\n"
        f"  pin provenance from `experiments/REFERENCES.json`.\n\n"
        f"This bundle contains **no** citizen weather-station data, **no** candidate models, and **no**\n"
        f"data from the sealed test split (2025 onward).\n\n"
        f"## Data Sources and License\n\n"
        f"- **Upstream sources (public domain):**\n"
        f"  - NOAA NCEI Integrated Surface Database (ISD) Global Hourly station observations\n"
        f"    (U.S. federal government work; public domain under 17 U.S.C. 105).\n"
        f"  - NOAA Global Ensemble Forecast System (GEFS v12) forecasts via NOAA Open Data\n"
        f"    Dissemination (U.S. federal government work; public domain under 17 U.S.C. 105).\n"
        f"- **License for derived files (`{RECOMMENDED_LICENSE}`):**\n"
        f"  We recommend **CC0-1.0 (Creative Commons Zero 1.0 Universal Public Domain Dedication)**\n"
        f"  for this derived bundle. The upstream NOAA datasets are U.S. federal public-domain works,\n"
        f"  and these files are deterministic numeric extractions (nearest-hour matching, station-point\n"
        f"  interpolation, and empirical quantiles). CC0 avoids attribution-stacking friction in\n"
        f"  automated verification and CI pipelines while the Zenodo DOI provides standard scholarly\n"
        f"  citation metadata. (CC-BY-4.0 is an acceptable alternative if formal attribution is desired.)\n\n"
        f"## Verification and Unpacking\n\n"
        f"Use `scripts/fetch_scoring_data.py` from the `wrfcoin/wrfcomp` repository to download,\n"
        f"verify against an expected archive digest or manifest plus `experiments/REFERENCES.json`,\n"
        f"and unpack into `$WRFCOMP_DATA_DIR` (default `~/wrfcomp-data`):\n\n"
        f"```bash\n"
        f"uv run python -P scripts/fetch_scoring_data.py \\\n"
        f"  --url <ZENODO_RECORD_URL> \\\n"
        f"  --expected-sha256 <ARCHIVE_SHA256> \\\n"
        f"  --year {year}\n"
        f"```\n"
    )


def _add_bytes_member(tf: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tf.addfile(info, io.BytesIO(payload))


def _add_verified_file_member(
    tf: tarfile.TarFile,
    data_dir: Path,
    relpath: str,
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Stream a regular file with ``O_NOFOLLOW`` and verify the exact bytes written to tar."""
    source = _assert_source_contained(data_dir, relpath, f"archive source {relpath}")
    fh, st = _open_regular_nofollow(source, f"archive source {relpath}")
    with fh:
        if st.st_size != expected_size:
            raise BundleError(
                f"Source file changed before packing ({relpath}): "
                f"size {st.st_size} != expected {expected_size}"
            )
        info = tarfile.TarInfo(name=relpath)
        info.size = expected_size
        info.mode = 0o644
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        reader = _HashingReader(fh)
        tf.addfile(info, reader)
        if reader.bytes_read != expected_size or fh.read(1):
            raise BundleError(f"Source file length changed while packing ({relpath})")
        if reader.hexdigest() != expected_sha256:
            raise BundleError(
                f"Source file bytes changed while packing ({relpath}): "
                f"packed {reader.hexdigest()}, expected {expected_sha256}"
            )


def _verify_written_archive(
    archive_path: Path,
    expected_members: list[tuple[str, int, str]],
) -> None:
    """Read back the finished archive and verify plain USTAR regular files and exact byte SHA-256."""
    seen: list[tuple[str, int, str]] = []
    with tarfile.open(archive_path, mode="r:gz") as tf:
        for member in tf:
            if member.pax_headers:
                raise BundleError(f"Written archive contains PAX headers on member: {member.name!r}")
            if not member.isfile() or member.type != tarfile.REGTYPE:
                raise BundleError(f"Written archive contains non-file member: {member.name!r}")
            src = tf.extractfile(member)
            if src is None:
                raise BundleError(f"Cannot read back written archive member: {member.name!r}")
            h = hashlib.sha256()
            nbytes = 0
            with src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
                    nbytes += len(chunk)
            seen.append((member.name, nbytes, h.hexdigest()))
            tf.members = []
    if seen != expected_members:
        raise BundleError("Archive read-back verification failed: packed bytes do not match manifest")


def build_bundle(
    *,
    year: int,
    data_dir: Path,
    out_dir: Path,
    references_path: Path = DEFAULT_REFERENCES,
    config_path: Path = DEFAULT_CONFIG,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    compress_level: int = DEFAULT_COMPRESS_LEVEL,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
) -> dict[str, Any]:
    """Verify all pinned artifacts for ``year`` and pack a reproducible ``.tar.gz`` archive."""
    cfg = load_config(config_path)
    _check_free_space(out_dir, min_free_gb)

    inventory, periods, items = collect_pinned_items(references_path, year, cfg)
    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()

    manifest_entries: list[dict[str, Any]] = []
    total_input_bytes = 0
    for item in items:
        target = _assert_source_contained(data_dir, item.relpath, f"{item.kind} {item.relpath}")
        st = os.lstat(target)
        if stat.S_ISDIR(st.st_mode):
            fmt, size_bytes, files, expected_var_names = _validate_zarr_store(
                data_dir,
                item.relpath,
                cfg=cfg,
                expected_model_id=item.expected_model_id,
                max_chunk_bytes=max_chunk_bytes,
                max_dataset_bytes=max_dataset_bytes,
            )
        elif stat.S_ISREG(st.st_mode):
            fmt, size_bytes, files, expected_var_names = _validate_netcdf_file(
                data_dir, item.relpath, kind=item.kind, cfg=cfg
            )
        else:
            raise BundleError(f"{item.relpath}: unsupported filesystem entry type")

        with _open_dataset(target, fmt, f"{item.kind} {item.relpath}") as ds:
            _check_dataset_bounds(ds, item.relpath, max_dataset_bytes=max_dataset_bytes)
            _check_not_sealed(ds, item.kind, item.relpath, cfg)
            try:
                if item.kind == "truth":
                    validate_truth_dataset(ds, cfg=cfg)
                elif item.kind == "thresholds":
                    validate_thresholds(ds)
                elif item.kind == "reference_forecast":
                    validate_forecast_dataset(ds, cfg=cfg)
            except SchemaError as exc:
                raise BundleError(f"{item.relpath}: harness schema validation failed: {exc}") from exc
            ds_vars = {str(name) for name in ds.variables}
            if ds_vars != expected_var_names:
                raise BundleError(
                    f"{item.relpath}: container datasets {sorted(expected_var_names)} "
                    f"do not match xarray variables {sorted(ds_vars)}"
                )
            actual = content_sha256(ds)
        if actual != item.expected_sha256:
            raise BundleError(
                f"Content hash mismatch for {item.relpath}: "
                f"computed {actual}, pinned {item.expected_sha256}"
            )
        total_input_bytes += size_bytes
        manifest_entries.append(
            {
                "content_sha256": actual,
                "files": files,
                "format": fmt,
                "kind": item.kind,
                "path": item.relpath,
                "pinned_by": list(item.pinned_by),
                "size_bytes": size_bytes,
            }
        )

    _check_free_space(out_dir, min_free_gb, required_extra_bytes=total_input_bytes)

    manifest = {
        "entries": manifest_entries,
        "harness_version": inventory.get("harness_version", {}),
        "license": RECOMMENDED_LICENSE,
        "manifest_version": MANIFEST_VERSION,
        "periods": periods,
        "protocol_version": inventory.get("protocol_version", ""),
        "references_sha256": periods_sha256(year_periods(inventory, year)),
        "total_size_bytes": total_input_bytes,
        "year": year,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    readme_bytes = render_bundle_readme(
        year=year,
        periods=periods,
        entries=manifest_entries,
        harness_version=manifest["harness_version"],
        protocol_version=manifest["protocol_version"],
    ).encode("utf-8")

    expected_members: list[tuple[str, int, str]] = [
        ("MANIFEST.json", len(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest()),
        ("README.md", len(readme_bytes), hashlib.sha256(readme_bytes).hexdigest()),
    ]
    for entry in manifest_entries:
        for fmeta in entry["files"]:
            expected_members.append((fmeta["path"], int(fmeta["size_bytes"]), str(fmeta["sha256"])))

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"wrfcomp-scoring-{year}.tar.gz"
    archive_path = out_dir / archive_name
    tmp_path = out_dir / f".{archive_name}.tmp"

    try:
        with tmp_path.open("wb") as raw_fh:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_fh, compresslevel=compress_level, mtime=0
            ) as gz_fh:
                with tarfile.open(fileobj=gz_fh, mode="w", format=tarfile.USTAR_FORMAT) as tf:
                    _add_bytes_member(tf, "MANIFEST.json", manifest_bytes)
                    _add_bytes_member(tf, "README.md", readme_bytes)
                    for entry in manifest_entries:
                        for fmeta in entry["files"]:
                            _add_verified_file_member(
                                tf,
                                data_dir,
                                fmeta["path"],
                                int(fmeta["size_bytes"]),
                                str(fmeta["sha256"]),
                            )
        _verify_written_archive(tmp_path, expected_members)
        tmp_path.replace(archive_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    archive_sha256 = _sha256_file(archive_path)
    archive_size = archive_path.stat().st_size

    sha_sidecar = out_dir / f"{archive_name}.sha256"
    sha_sidecar.write_text(f"{archive_sha256}  {archive_name}\n", encoding="utf-8")
    manifest_sidecar = out_dir / f"wrfcomp-scoring-{year}.MANIFEST.json"
    manifest_sidecar.write_bytes(manifest_bytes)

    return {
        "archive": archive_name,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size,
        "entries": len(manifest_entries),
        "manifest_sidecar": str(manifest_sidecar),
        "periods": periods,
        "sha256_sidecar": str(sha_sidecar),
        "uncompressed_size_bytes": total_input_bytes,
        "year": year,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a verified per-year scoring data bundle.")
    parser.add_argument("--year", type=int, default=2024, help="Calendar year to bundle (default: 2024).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="WRFComp data directory (default: $WRFCOMP_DATA_DIR or ~/wrfcomp-data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for the archive (default: <data-dir>/bundle).",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=DEFAULT_REFERENCES,
        help="Path to experiments/REFERENCES.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to configs/splits.yaml.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=f"Refuse to pack if free disk space is below this GiB floor (default: {DEFAULT_MIN_FREE_GB}).",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=DEFAULT_COMPRESS_LEVEL,
        choices=range(1, 10),
        help=f"Gzip compression level 1-9 (default: {DEFAULT_COMPRESS_LEVEL}).",
    )
    args = parser.parse_args(argv)
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    out_dir = args.out_dir if args.out_dir is not None else data_dir / "bundle"
    try:
        summary = build_bundle(
            year=args.year,
            data_dir=data_dir,
            out_dir=out_dir,
            references_path=args.references,
            config_path=args.config,
            min_free_gb=args.min_free_gb,
            compress_level=args.compress_level,
        )
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "archive": summary["archive"],
                "archive_sha256": summary["archive_sha256"],
                "archive_size_bytes": summary["archive_size_bytes"],
                "entries": summary["entries"],
                "periods": summary["periods"],
                "uncompressed_size_bytes": summary["uncompressed_size_bytes"],
                "year": summary["year"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
