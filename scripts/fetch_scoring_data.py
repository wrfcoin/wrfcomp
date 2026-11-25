"""Download, verify, and unpack a WRFComp scoring data bundle from a Zenodo record URL.

No Zenodo record URL is hard-coded. Pass ``--url <ZENODO_RECORD_OR_FILE_URL>`` (or
``--archive <LOCAL_ARCHIVE>`` for offline use) together with an independently trusted physical
pin (``--expected-sha256``, ``--expected-sha256-file``, or ``--expected-manifest``).

Extraction streams members with bounded download size, member count, per-member size, manifest
size, total expanded bytes, and staging disk space. Every unpacked dataset is validated against
strict Zarr v2 / NetCDF layout rules, bounded decoded size, per-file byte SHA-256 in
``MANIFEST.json``, ``references_sha256``, and the harness ``content_sha256`` pins in
``experiments/REFERENCES.json`` before symlink-checked atomic installation into ``--data-dir``.

Usage:
    uv run python -P scripts/fetch_scoring_data.py \
      --url <ZENODO_RECORD_URL> --expected-sha256 <SHA256> --year 2024
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PureWindowsPath
from typing import Any

import h5py
import numpy as np
import pandas as pd
import xarray as xr

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
DEFAULT_MAX_ARCHIVE_BYTES = 8 * 1024**3
DEFAULT_MAX_MEMBERS = 50_000
DEFAULT_MAX_MEMBER_BYTES = 512 * 1024**2
DEFAULT_MAX_TOTAL_BYTES = 10 * 1024**3
DEFAULT_MAX_TAR_HEADERS = 50_000
DEFAULT_MAX_TAR_METADATA_BYTES = 64 * 1024**2
DEFAULT_MAX_MANIFEST_BYTES = 16 * 1024**2
DEFAULT_MAX_README_BYTES = 1 * 1024**2
DEFAULT_MAX_METADATA_BYTES = 1 * 1024**2
DEFAULT_MAX_CHUNK_BYTES = 256 * 1024**2
DEFAULT_MAX_DATASET_BYTES = 2 * 1024**3
DEFAULT_MIN_FREE_BYTES = 256 * 1024**2
MANIFEST_VERSION = 1
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
PERIOD_RE = re.compile(r"^(?P<year>[0-9]{4})-(?:[0-9]{2}|Q[1-4])$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK_INDEX_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
ZARR_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
ZENODO_RECORD_RE = re.compile(r"^/(?:records?|api/records)/(?P<id>[0-9]+)/?$")
ZENODO_DOI_RE = re.compile(r"^/10\.5281/zenodo\.(?P<id>[0-9]+)/?$")
NETCDF_MAGICS = (b"\x89HDF\r\n\x1a\n", b"CDF\x01", b"CDF\x02")
MAX_ATTR_VALUE_LEN = 512

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
_TAR_ZERO_BLOCK = b"\x00" * 512
_TAR_EXTENDED_TYPEFLAGS = frozenset({b"x", b"g", b"X", b"L", b"K", b"S"})
_TAR_ALLOWED_TYPEFLAGS = frozenset({b"0", b"\x00", b"5"})
_USTAR_MAGIC = b"ustar\x00"
_USTAR_VERSION = b"00"


class FetchError(RuntimeError):
    """Raised when downloading, verifying, or unpacking a scoring bundle fails."""


def _parse_ustar_octal_size(field: bytes) -> int:
    if not field:
        return 0
    if field[0] & 0x80:
        raise FetchError("Refusing GNU base-256 tar size encoding; plain USTAR format only")
    try:
        text = field.rstrip(b"\x00 ").decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise FetchError(f"Non-ASCII tar header size field: {exc}") from exc
    if not text:
        return 0
    try:
        val = int(text, 8)
    except ValueError as exc:
        raise FetchError(f"Malformed USTAR octal size field: {text!r}") from exc
    if val < 0:
        raise FetchError("Negative size in tar header")
    return val


class _TarBudgetStream:
    """Plain USTAR stream wrapper enforcing strict framing and total decompressed byte budgets.

    Only POSIX USTAR regular files (``0``/``\\x00``) and directories (``5``) with ASCII octal
    sizes are permitted. PAX and GNU extended headers (``x``, ``g``, ``X``, ``L``, ``K``, ``S``)
    and all non-regular/non-directory member types are rejected immediately on the 512-byte
    header before ``tarfile`` reads any record body, guaranteeing identical record boundaries
    between this wrapper and ``tarfile.open(..., mode="r|")``.

    Every byte read from the decompressed stream is strictly bounded by
    ``max_total_bytes + max_metadata_bytes``, with tar headers/padding/EOF blocks also capped
    independently by ``max_headers`` and ``max_metadata_bytes``.
    """

    def __init__(
        self,
        stream: Any,
        *,
        max_headers: int,
        max_metadata_bytes: int,
        max_member_bytes: int,
        max_total_bytes: int,
    ) -> None:
        self._stream = stream
        self._max_headers = max_headers
        self._max_metadata_bytes = max_metadata_bytes
        self._max_member_bytes = max_member_bytes
        self._max_total_bytes = max_total_bytes
        self._max_stream_bytes = max_total_bytes + max_metadata_bytes
        self._buf = bytearray()
        self._remaining_body = 0
        self.header_count = 0
        self.metadata_bytes = 0
        self.payload_bytes = 0
        self.total_stream_bytes = 0

    def _raw_read(self, n: int) -> bytes:
        piece = self._stream.read(n)
        if not piece:
            return b""
        self.total_stream_bytes += len(piece)
        if self.total_stream_bytes > self._max_stream_bytes:
            raise FetchError(
                f"Decompressed archive stream exceeded maximum byte budget ({self._max_stream_bytes} bytes)"
            )
        return piece

    def _read_exact(self, n: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < n:
            piece = self._raw_read(n - len(chunks))
            if not piece:
                break
            chunks.extend(piece)
        return bytes(chunks)

    def _charge_metadata(self, nbytes: int) -> None:
        self.metadata_bytes += nbytes
        if self.metadata_bytes > self._max_metadata_bytes:
            raise FetchError(
                f"Archive tar metadata bytes exceeded independent budget ({self._max_metadata_bytes} bytes)"
            )

    def _next_header_block(self) -> bytes:
        block = self._read_exact(512)
        if not block:
            return b""
        if len(block) < 512 or block == _TAR_ZERO_BLOCK:
            self._charge_metadata(len(block))
            return block

        self.header_count += 1
        if self.header_count > self._max_headers:
            raise FetchError(
                f"Archive tar header record count exceeded independent budget ({self._max_headers})"
            )
        self._charge_metadata(512)

        typeflag = block[156:157]
        if typeflag in _TAR_EXTENDED_TYPEFLAGS:
            raise FetchError(
                f"Refusing tar PAX/GNU extended header (typeflag {typeflag!r}); "
                "plain USTAR regular files and directories only"
            )
        if typeflag not in _TAR_ALLOWED_TYPEFLAGS:
            raise FetchError(
                f"Refusing non-regular tar member typeflag {typeflag!r}; "
                "plain USTAR regular files and directories only"
            )
        if block[257:263] != _USTAR_MAGIC or block[263:265] != _USTAR_VERSION:
            raise FetchError("Refusing non-USTAR tar header; plain USTAR format only")

        size = _parse_ustar_octal_size(block[124:136])
        if typeflag == b"5" and size != 0:
            raise FetchError(f"Refusing tar directory header with non-zero size ({size})")
        if size > self._max_member_bytes:
            raise FetchError(
                f"Archive member size {size} exceeds per-member limit ({self._max_member_bytes})"
            )
        if self.payload_bytes + size > self._max_total_bytes:
            raise FetchError(
                f"Archive expanded payload exceeds limit ({self._max_total_bytes} bytes)"
            )
        padded = ((size + 511) // 512) * 512
        self._charge_metadata(padded - size)
        self.payload_bytes += size
        self._remaining_body = padded
        return block

    def read(self, size: int = -1) -> bytes:
        want = 65536 if (size is None or size < 0) else size
        if want == 0:
            return b""
        out = bytearray()
        while len(out) < want:
            if self._buf:
                take = min(want - len(out), len(self._buf))
                out.extend(self._buf[:take])
                del self._buf[:take]
            elif self._remaining_body > 0:
                take = min(want - len(out), self._remaining_body)
                piece = self._raw_read(take)
                if not piece:
                    break
                out.extend(piece)
                self._remaining_body -= len(piece)
            else:
                block = self._next_header_block()
                if not block:
                    break
                self._buf.extend(block)
        return bytes(out)


def default_data_dir() -> Path:
    env = os.environ.get("WRFCOMP_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "wrfcomp-data"


def _validate_relpath(relpath: str, what: str) -> Path:
    if not isinstance(relpath, str) or not relpath:
        raise FetchError(f"{what}: empty or non-string relative path")
    if "\x00" in relpath or any(ord(ch) < 32 for ch in relpath):
        raise FetchError(f"{what}: control characters are forbidden in path {relpath!r}")
    if relpath.startswith("/") or "\\" in relpath or ":" in relpath or "//" in relpath:
        raise FetchError(f"{what}: unsafe path syntax {relpath!r}")
    win = PureWindowsPath(relpath)
    if win.is_absolute() or win.drive:
        raise FetchError(f"{what}: Windows drive/absolute path forbidden: {relpath!r}")
    raw_parts = relpath.split("/")
    for part in raw_parts:
        if not part or part in (".", ".."):
            raise FetchError(f"{what}: unsafe path component {part!r} in {relpath!r}")
        if part.endswith(" ") or part.endswith("."):
            raise FetchError(f"{what}: Windows-unsafe trailing space/dot in {part!r}")
        stem = part.split(".", 1)[0].lower()
        if stem in WINDOWS_RESERVED_NAMES:
            raise FetchError(f"{what}: Windows reserved device name in {relpath!r}")
    p = Path(*raw_parts)
    lowered = {part.lower() for part in p.parts}
    forbidden = lowered & FORBIDDEN_PATH_PARTS
    if forbidden:
        raise FetchError(f"{what}: forbidden path component {sorted(forbidden)} in {relpath!r}")
    return p


def _open_regular_nofollow(path: Path, what: str) -> tuple[Any, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise FetchError(f"{what}: cannot safely open {path.name}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise FetchError(f"{what}: not a regular file: {path.name}")
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


def _md5_file(path: Path) -> str:
    fh, _ = _open_regular_nofollow(path, f"file {path.name}")
    h = hashlib.md5()
    with fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_bounded_bytes(path: Path, max_bytes: int, what: str) -> bytes:
    fh, st = _open_regular_nofollow(path, what)
    with fh:
        if st.st_size > max_bytes:
            raise FetchError(f"{what}: file exceeds {max_bytes} bytes")
        raw = fh.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise FetchError(f"{what}: file exceeds {max_bytes} bytes")
    return raw


def _read_bounded_json(path: Path, max_bytes: int, what: str) -> Any:
    raw = _read_bounded_bytes(path, max_bytes, what)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FetchError(f"{what}: invalid JSON: {exc}") from exc


def _assert_tree_no_symlinks(root: Path, relpath: str, what: str, *, must_exist: bool) -> Path:
    """Validate that every existing ancestor (and nested entry, if any) under ``root`` has no symlinks."""
    rel = _validate_relpath(relpath, what)
    root_real = root.resolve(strict=True)
    cur = root
    for idx, part in enumerate(rel.parts):
        cur = cur / part
        is_last = idx == len(rel.parts) - 1
        if not os.path.lexists(cur):
            if must_exist:
                raise FetchError(f"{what}: missing path {relpath}")
            continue
        st = os.lstat(cur)
        if stat.S_ISLNK(st.st_mode):
            raise FetchError(f"{what}: symlink refused at {cur.relative_to(root).as_posix()}")
        if not is_last and not stat.S_ISDIR(st.st_mode):
            raise FetchError(f"{what}: non-directory ancestor in {relpath}")
        if not cur.resolve().is_relative_to(root_real):
            raise FetchError(f"{what}: path escapes root directory: {relpath}")
    if os.path.lexists(cur) and stat.S_ISDIR(os.lstat(cur).st_mode):
        for dirpath, dirnames, filenames in os.walk(cur, followlinks=False):
            dp = Path(dirpath)
            for name in (*dirnames, *filenames):
                child = dp / name
                cst = os.lstat(child)
                if stat.S_ISLNK(cst.st_mode):
                    raise FetchError(
                        f"{what}: symlink refused inside {relpath}: {child.relative_to(cur).as_posix()}"
                    )
                if not (stat.S_ISREG(cst.st_mode) or stat.S_ISDIR(cst.st_mode)):
                    raise FetchError(f"{what}: non-regular entry inside {relpath}: {name!r}")
                if not child.resolve().is_relative_to(root_real):
                    raise FetchError(f"{what}: nested entry escapes root: {relpath}")
    return cur


def _as_text(val: Any, what: str) -> str:
    if isinstance(val, bytes | np.bytes_):
        try:
            text = bytes(val).decode("ascii")
        except UnicodeDecodeError as exc:
            raise FetchError(f"{what}: non-ASCII bytes value refused") from exc
    elif isinstance(val, str | np.str_):
        text = str(val)
    else:
        raise FetchError(f"{what}: expected string value, got {type(val).__name__}")
    if len(text) > MAX_ATTR_VALUE_LEN or "\x00" in text:
        raise FetchError(f"{what}: string too large or contains NUL")
    return text


def _as_scalar_float(val: Any, what: str) -> float:
    arr = np.asarray(val)
    if arr.size != 1 or arr.dtype.kind not in "iuf":
        raise FetchError(f"{what}: expected numeric scalar")
    out = float(arr.reshape(-1)[0])
    if not math.isfinite(out):
        raise FetchError(f"{what}: non-finite numeric value {out!r} refused")
    return out


def _as_scalar_int(val: Any, what: str) -> int:
    arr = np.asarray(val)
    if arr.size != 1 or arr.dtype.kind not in "iu":
        raise FetchError(f"{what}: expected integer scalar")
    return int(arr.reshape(-1)[0])


def _validate_train_end_value(val: Any, cfg: HarnessConfig, what: str) -> None:
    text = _as_text(val, what)
    if not ISO_TIMESTAMP_RE.fullmatch(text):
        raise FetchError(f"{what}: invalid ISO timestamp {text!r}")
    assert cfg.splits.train.end is not None
    if pd.Timestamp(text) != pd.Timestamp(cfg.splits.train.end):
        raise FetchError(
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
                raise FetchError(
                    f"{what}: invalid units {text!r} for {var_name} (expected one of {sorted(allowed)})"
                )
        elif var_name.endswith(OFFSET_SUFFIX) and var_name[: -len(OFFSET_SUFFIX)] in VARIABLE_UNITS:
            if text != "minutes":
                raise FetchError(f"{what}: invalid units {text!r} for offset {var_name}")
        elif var_name == "elevation_m":
            if text != "m":
                raise FetchError(f"{what}: invalid units {text!r} for elevation_m")
        elif var_name == "latitude":
            if text != "degrees_north":
                raise FetchError(f"{what}: invalid units {text!r} for latitude")
        elif var_name == "longitude":
            if text != "degrees_east":
                raise FetchError(f"{what}: invalid units {text!r} for longitude")
        elif var_name in ("valid_time", "init_time"):
            if not CF_TIME_UNITS_RE.fullmatch(text):
                raise FetchError(f"{what}: invalid CF time units {text!r}")
        elif var_name == "lead_time":
            if text not in APPROVED_LEAD_UNITS:
                raise FetchError(f"{what}: invalid lead_time units {text!r}")
        else:
            raise FetchError(f"{what}: unexpected units attribute on {var_name!r}")
        return

    if attr_name == "long_name":
        if kind != "truth" or not var_name.endswith(OFFSET_SUFFIX):
            raise FetchError(f"{what}: long_name is only approved on truth offset variables")
        base = var_name[: -len(OFFSET_SUFFIX)]
        expected = f"Observation time offset from valid_time for {base} (-32768 when missing)"
        text = _as_text(val, what)
        if text != expected:
            raise FetchError(f"{what}: unapproved long_name {text!r}")
        return

    if attr_name == "calendar":
        if var_name not in ("valid_time", "init_time"):
            raise FetchError(f"{what}: calendar only approved on time coordinates")
        text = _as_text(val, what)
        if text not in APPROVED_CALENDARS:
            raise FetchError(f"{what}: unapproved calendar {text!r}")
        return

    if attr_name == "coordinates":
        text = _as_text(val, what)
        tokens = text.split()
        if not tokens or not set(tokens).issubset(APPROVED_COORD_TOKENS):
            raise FetchError(f"{what}: unapproved coordinates attribute {text!r}")
        return

    if attr_name == "tail_quantile":
        if kind != "thresholds":
            raise FetchError(f"{what}: tail_quantile only approved on thresholds")
        q = _as_scalar_float(val, what)
        expected_q = float(cfg.gate.tail_quantile)
        if not (math.isfinite(q) and math.isfinite(expected_q) and q == expected_q):
            raise FetchError(f"{what}: tail_quantile {q!r} != {expected_q!r}")
        return

    if attr_name == "train_end_utc":
        if kind != "thresholds":
            raise FetchError(f"{what}: train_end_utc only approved on thresholds")
        _validate_train_end_value(val, cfg, what)
        return

    if attr_name == "_ARRAY_DIMENSIONS":
        ok_dims = isinstance(val, list) and bool(val) and all(
            isinstance(d, str) and d in APPROVED_DIMS for d in val
        )
        if not ok_dims:
            raise FetchError(f"{what}: unapproved _ARRAY_DIMENSIONS {val!r}")
        return

    if attr_name == "dtype":
        text = _as_text(val, what)
        if var_name != "lead_time" or text != "timedelta64[ns]":
            raise FetchError(f"{what}: unapproved dtype attribute {text!r} on {var_name!r}")
        return

    if attr_name == "CLASS":
        text = _as_text(val, what)
        if text != "DIMENSION_SCALE":
            raise FetchError(f"{what}: unapproved HDF5 CLASS {text!r}")
        return

    if attr_name == "NAME":
        text = _as_text(val, what)
        if text != var_name and not text.startswith("This is a netCDF dimension"):
            raise FetchError(f"{what}: unapproved HDF5 NAME {text!r}")
        return

    if attr_name in (
        "_Netcdf4Dimid",
        "_Netcdf4Coordinates",
        "_FillValue",
        "DIMENSION_LIST",
        "REFERENCE_LIST",
    ):
        return

    raise FetchError(f"{what}: unapproved attribute {attr_name!r}")


def _validate_netcdf_root_attr_value(
    attr_name: str, val: Any, *, kind: str, cfg: HarnessConfig, what: str
) -> None:
    if attr_name == "_NCProperties":
        text = _as_text(val, what)
        if not NCPROPERTIES_RE.fullmatch(text):
            raise FetchError(f"{what}: unapproved _NCProperties value {text!r}")
        return
    if attr_name == "protocol_version":
        text = _as_text(val, what)
        if text != cfg.protocol_version:
            raise FetchError(f"{what}: protocol_version {text!r} != {cfg.protocol_version!r}")
        return
    if kind == "truth":
        if attr_name == "source":
            text = _as_text(val, what)
            if text not in APPROVED_TRUTH_SOURCES:
                raise FetchError(f"{what}: unapproved truth source {text!r}")
            return
        if attr_name == "parser_version":
            text = _as_text(val, what)
            if not PARSER_VERSION_RE.fullmatch(text):
                raise FetchError(f"{what}: invalid parser_version {text!r}")
            return
        if attr_name == "truth_tolerance_minutes":
            tol = _as_scalar_int(val, what)
            if tol != int(cfg.evaluation.truth_tolerance_minutes):
                raise FetchError(
                    f"{what}: truth_tolerance_minutes {tol} != {cfg.evaluation.truth_tolerance_minutes}"
                )
            return
        if attr_name == "passing_qc_codes":
            text = _as_text(val, what)
            if text != passing_qc_signature():
                raise FetchError(f"{what}: unapproved passing_qc_codes {text!r}")
            return
    if kind == "thresholds":
        if attr_name == "tail_quantile":
            q = _as_scalar_float(val, what)
            expected_q = float(cfg.gate.tail_quantile)
            if not (math.isfinite(q) and math.isfinite(expected_q) and q == expected_q):
                raise FetchError(f"{what}: tail_quantile {q!r} != {expected_q!r}")
            return
        if attr_name == "train_end_utc":
            _validate_train_end_value(val, cfg, what)
            return
    raise FetchError(f"{what}: unapproved root attribute {attr_name!r}")


def _validate_zarr_root_attrs(
    attrs: Any,
    *,
    expected_model_id: str | None,
    what: str,
) -> dict[str, Any]:
    if not isinstance(attrs, dict):
        raise FetchError(f"{what}: must be a JSON object")
    disallowed = sorted(set(attrs) - ALLOWED_ZARR_ROOT_ATTRS)
    if disallowed:
        raise FetchError(f"{what}: unapproved metadata attributes refused: {disallowed}")
    model_id = _as_text(attrs.get("model_id"), f"{what}['model_id']")
    if expected_model_id is not None and model_id != expected_model_id:
        raise FetchError(f"{what}: model_id {model_id!r} != pinned {expected_model_id!r}")
    if model_id not in REQUIRED_BASELINES:
        raise FetchError(f"{what}: unapproved reference model_id {model_id!r}")
    model_ver = _as_text(attrs.get("model_version"), f"{what}['model_version']")
    if not MODEL_VERSION_RE.fullmatch(model_ver):
        raise FetchError(f"{what}: unapproved model_version {model_ver!r}")
    return attrs


def _validate_zarr_var_attrs(
    attrs: Any,
    *,
    var_name: str,
    cfg: HarnessConfig,
    what: str,
) -> dict[str, Any]:
    if not isinstance(attrs, dict):
        raise FetchError(f"{what}: must be a JSON object")
    disallowed = sorted(set(attrs) - ALLOWED_ZARR_VAR_ATTRS)
    if disallowed:
        raise FetchError(f"{what}: unapproved metadata attributes refused: {disallowed}")
    for key, val in attrs.items():
        _validate_var_attr_value(
            var_name, key, val, kind="reference_forecast", cfg=cfg, what=f"{what}[{key!r}]"
        )
    return attrs


def _validate_zarr_store(
    root: Path,
    relpath: str,
    *,
    cfg: HarnessConfig,
    expected_model_id: str | None = None,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
    max_metadata_bytes: int = DEFAULT_MAX_METADATA_BYTES,
) -> set[str]:
    """Validate strict Zarr v2 layout, approved metadata, and chunk/dataset bounds BEFORE xarray."""
    store = _assert_tree_no_symlinks(root, relpath, f"zarr store {relpath}", must_exist=True)
    if not stat.S_ISDIR(os.lstat(store).st_mode):
        raise FetchError(f"{relpath}: expected a Zarr store directory")

    root_files: dict[str, Path] = {}
    var_dirs: dict[str, Path] = {}
    for child in sorted(store.iterdir(), key=lambda p: p.name):
        st = os.lstat(child)
        if stat.S_ISREG(st.st_mode):
            if child.name not in (".zgroup", ".zattrs", ".zmetadata"):
                raise FetchError(f"{relpath}: unexpected file at Zarr store root: {child.name!r}")
            root_files[child.name] = child
        elif stat.S_ISDIR(st.st_mode):
            if not ZARR_VAR_NAME_RE.fullmatch(child.name):
                raise FetchError(f"{relpath}: unexpected subdirectory in Zarr store: {child.name!r}")
            var_dirs[child.name] = child
        else:
            raise FetchError(f"{relpath}: non-regular entry in Zarr store: {child.name!r}")

    if ".zgroup" not in root_files:
        raise FetchError(f"{relpath}: missing .zgroup in Zarr v2 store")
    zgroup = _read_bounded_json(root_files[".zgroup"], max_metadata_bytes, f"{relpath}/.zgroup")
    if (
        not isinstance(zgroup, dict)
        or set(zgroup) != ALLOWED_ZARR_ZGROUP_KEYS
        or zgroup.get("zarr_format") != 2
    ):
        raise FetchError(f"{relpath}: only Zarr v2 groups with standard keys are supported")

    parsed_metadata: dict[str, Any] = {".zgroup": zgroup}
    if ".zattrs" in root_files:
        zattrs = _validate_zarr_root_attrs(
            _read_bounded_json(root_files[".zattrs"], max_metadata_bytes, f"{relpath}/.zattrs"),
            expected_model_id=expected_model_id,
            what=f"{relpath}/.zattrs",
        )
        parsed_metadata[".zattrs"] = zattrs
    else:
        raise FetchError(f"{relpath}: missing .zattrs with model_id/model_version")

    if not var_dirs:
        raise FetchError(f"{relpath}: Zarr store contains no arrays")

    total_decoded_bytes = 0
    for var_name, vdir in sorted(var_dirs.items()):
        zarray_path = vdir / ".zarray"
        if not os.path.lexists(zarray_path) or not stat.S_ISREG(os.lstat(zarray_path).st_mode):
            raise FetchError(f"{relpath}/{var_name}: missing regular .zarray metadata")
        za = _read_bounded_json(zarray_path, max_metadata_bytes, f"{relpath}/{var_name}/.zarray")
        if (
            not isinstance(za, dict)
            or not set(za).issubset(ALLOWED_ZARR_ZARRAY_KEYS)
            or za.get("zarr_format") != 2
        ):
            raise FetchError(f"{relpath}/{var_name}: unsupported .zarray format or keys")
        shape = za.get("shape")
        chunks = za.get("chunks")
        dtype_raw = za.get("dtype")
        sep = za.get("dimension_separator", ".")
        if sep not in (".", "/"):
            raise FetchError(f"{relpath}/{var_name}: unsupported dimension_separator {sep!r}")
        if (
            not isinstance(shape, list)
            or not isinstance(chunks, list)
            or len(shape) != len(chunks)
            or any(not isinstance(s, int) or s < 0 for s in shape)
            or any(not isinstance(c, int) or c <= 0 for c in chunks)
        ):
            raise FetchError(f"{relpath}/{var_name}: invalid shape/chunks in .zarray")
        try:
            itemsize = max(int(np.dtype(dtype_raw).itemsize), 8)
        except Exception as exc:
            raise FetchError(f"{relpath}/{var_name}: invalid dtype {dtype_raw!r}") from exc
        chunk_elems = math.prod(chunks) if chunks else 1
        array_elems = math.prod(shape) if shape else 1
        if chunk_elems * itemsize > max_chunk_bytes:
            raise FetchError(
                f"{relpath}/{var_name}: chunk size {chunk_elems * itemsize} exceeds limit {max_chunk_bytes}"
            )
        total_decoded_bytes += array_elems * itemsize
        if total_decoded_bytes > max_dataset_bytes:
            raise FetchError(f"{relpath}: decoded dataset size exceeds limit {max_dataset_bytes} bytes")

        parsed_metadata[f"{var_name}/.zarray"] = za
        grid_limits = [math.ceil(s / c) for s, c in zip(shape, chunks, strict=True)] if shape else [1]

        for dirpath, dirnames, filenames in os.walk(vdir, followlinks=False):
            cur_dir = Path(dirpath)
            if cur_dir != vdir and sep != "/":
                raise FetchError(
                    f"{relpath}/{var_name}: unexpected subdirectory {cur_dir.relative_to(vdir).as_posix()!r}"
                )
            for dname in dirnames:
                if sep != "/" or not CHUNK_INDEX_RE.fullmatch(dname):
                    raise FetchError(f"{relpath}/{var_name}: unexpected directory {dname!r}")
            for fname in filenames:
                fpath = cur_dir / fname
                inner_rel = fpath.relative_to(vdir).as_posix()
                if inner_rel == ".zarray":
                    continue
                if inner_rel == ".zattrs":
                    vattrs = _validate_zarr_var_attrs(
                        _read_bounded_json(fpath, max_metadata_bytes, f"{relpath}/{var_name}/.zattrs"),
                        var_name=var_name,
                        cfg=cfg,
                        what=f"{relpath}/{var_name}/.zattrs",
                    )
                    parsed_metadata[f"{var_name}/.zattrs"] = vattrs
                    continue
                parts = inner_rel.split(sep)
                if len(parts) != len(grid_limits):
                    raise FetchError(
                        f"{relpath}/{var_name}: unrelated file or malformed chunk key {inner_rel!r}"
                    )
                for part, lim in zip(parts, grid_limits, strict=True):
                    if not CHUNK_INDEX_RE.fullmatch(part) or int(part) >= lim:
                        raise FetchError(
                            f"{relpath}/{var_name}: out-of-bounds or unrelated file {inner_rel!r}"
                        )

    if ".zmetadata" in root_files:
        zmeta = _read_bounded_json(root_files[".zmetadata"], max_metadata_bytes, f"{relpath}/.zmetadata")
        if (
            not isinstance(zmeta, dict)
            or set(zmeta) != ALLOWED_ZARR_ZMETADATA_KEYS
            or zmeta.get("zarr_consolidated_format") != 1
            or not isinstance(zmeta.get("metadata"), dict)
        ):
            raise FetchError(f"{relpath}/.zmetadata is malformed")
        meta_map = zmeta["metadata"]
        if set(meta_map) != set(parsed_metadata):
            raise FetchError(f"{relpath}/.zmetadata keys do not match on-disk Zarr metadata files")
        for mkey, mval in meta_map.items():
            if mval != parsed_metadata[mkey]:
                raise FetchError(f"{relpath}/.zmetadata disagrees with on-disk {mkey}")

    return set(var_dirs)


def _validate_netcdf_file(root: Path, relpath: str, *, kind: str, cfg: HarnessConfig) -> set[str]:
    target = _assert_tree_no_symlinks(root, relpath, f"netcdf {relpath}", must_exist=True)
    if not stat.S_ISREG(os.lstat(target).st_mode):
        raise FetchError(f"{relpath}: expected a regular NetCDF file")
    fh, _ = _open_regular_nofollow(target, f"netcdf {relpath}")
    with fh:
        head = fh.read(8)
    if head != NETCDF_MAGICS[0]:
        raise FetchError(f"{relpath}: expected an HDF5/NetCDF4 file")

    allowed_root = ALLOWED_HDF5_ROOT_ATTRS.get(kind)
    allowed_var = ALLOWED_HDF5_VAR_ATTRS.get(kind)
    if allowed_root is None or allowed_var is None:
        raise FetchError(f"{relpath}: unsupported NetCDF dataset kind {kind!r}")

    try:
        with h5py.File(target, "r") as h5f:
            disallowed_root = sorted(set(h5f.attrs.keys()) - allowed_root)
            if disallowed_root:
                raise FetchError(
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
                    raise FetchError(f"{relpath}: non-hardlink HDF5 entry refused: {name!r}")
                obj = h5f[name]
                if isinstance(obj, h5py.Group):
                    raise FetchError(f"{relpath}: nested NetCDF/HDF5 group refused: {name!r}")
                if not isinstance(obj, h5py.Dataset):
                    raise FetchError(f"{relpath}: unsupported HDF5 object refused: {name!r}")
                disallowed_var = sorted(set(obj.attrs.keys()) - allowed_var)
                if disallowed_var:
                    raise FetchError(
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
    except FetchError:
        raise
    except Exception as exc:
        raise FetchError(f"{relpath}: failed to inspect HDF5/NetCDF4 structure: {exc}") from exc
    return hdf5_datasets


def _open_dataset(path: Path, fmt: str, what: str) -> xr.Dataset:
    try:
        if fmt == "zarr":
            try:
                return xr.open_zarr(path, chunks=None, consolidated=True)
            except (KeyError, ValueError, FileNotFoundError):
                return xr.open_zarr(path, chunks=None, consolidated=False)
        return xr.open_dataset(path)
    except Exception as exc:
        raise FetchError(f"{what}: failed to open {path.name}: {exc}") from exc


def _check_dataset_bounds(
    ds: xr.Dataset,
    relpath: str,
    *,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
) -> None:
    total = 0
    for vname, var in ds.variables.items():
        if any(s < 0 for s in var.shape):
            raise FetchError(f"{relpath}: negative dimension on {vname}")
        itemsize = max(int(var.dtype.itemsize), 8)
        total += int(var.size) * itemsize
        if total > max_dataset_bytes:
            raise FetchError(f"{relpath}: decoded dataset size exceeds limit {max_dataset_bytes} bytes")


def _check_not_sealed(ds: xr.Dataset, kind: str, relpath: str, cfg: HarnessConfig) -> None:
    sealed_start = cfg.splits.sealed_test.start
    if kind == "truth":
        if "valid_time" not in ds.coords:
            raise FetchError(f"{relpath}: truth dataset missing valid_time coordinate")
        check_not_sealed(ds["valid_time"].values, what=f"truth {relpath}", sealed_start=sealed_start)
    elif kind == "reference_forecast":
        check_forecast_not_sealed(ds, what=f"reference {relpath}", sealed_start=sealed_start)


def expected_pins(
    references_path: Path,
    year: int,
    cfg: HarnessConfig,
) -> tuple[list[str], dict[str, tuple[str, str, str | None]]]:
    """Return ``(periods, {relpath: (kind, expected_content_sha256, model_id)})`` from ``REFERENCES.json``."""
    sealed_dt = cfg.splits.sealed_test.start_utc
    if year >= sealed_dt.year:
        raise FetchError(
            f"Refusing year {year}: sealed split starts at {sealed_dt.replace(tzinfo=None).isoformat()}"
        )
    try:
        inventory = json.loads(references_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FetchError(f"Cannot read references inventory: {exc}") from exc
    if not isinstance(inventory, dict) or not isinstance(inventory.get("periods"), dict):
        raise FetchError("Invalid REFERENCES.json structure")

    periods: list[str] = []
    pins: dict[str, tuple[str, str, str | None]] = {}

    def record(relpath: str, kind: str, digest: str, model_id: str | None = None) -> None:
        _validate_relpath(relpath, f"REFERENCES.json {kind}")
        if relpath in pins:
            if pins[relpath] != (kind, digest, model_id):
                raise FetchError(f"Conflicting pins in REFERENCES.json for {relpath}")
        else:
            pins[relpath] = (kind, digest, model_id)

    for period in sorted(inventory["periods"]):
        m = PERIOD_RE.fullmatch(period)
        if not m or int(m.group("year")) != year:
            continue
        periods.append(period)
        entry = inventory["periods"][period]
        refs = entry.get("references")
        if not isinstance(refs, dict) or set(refs) != set(REQUIRED_BASELINES):
            raise FetchError(f"Period {period} must pin exactly {', '.join(REQUIRED_BASELINES)}")
        for baseline in REQUIRED_BASELINES:
            rec = refs[baseline]
            record(entry["truth"], "truth", rec["truth_sha256"])
            record(entry["thresholds"], "thresholds", rec["thresholds_sha256"])
            record(rec["store"], "reference_forecast", rec["forecast_sha256"], str(rec["model_id"]))

    if not periods:
        raise FetchError(f"No periods pinned for year {year} in {references_path.name}")
    return periods, pins


def _check_url_scheme(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https":
        return parsed
    if parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost"):
        return parsed
    raise FetchError(
        f"Refusing URL scheme {parsed.scheme!r}: use https:// (or --archive for a local file)"
    )


def resolve_download_url(url: str, year: int, *, timeout: float = 60.0) -> tuple[str, str | None]:
    """Resolve a Zenodo record URL (or direct archive URL) to ``(file_url, expected_md5)``."""
    parsed = _check_url_scheme(url)
    api_url: str | None = None
    host = (parsed.hostname or "").lower()
    if host in ("zenodo.org", "www.zenodo.org", "sandbox.zenodo.org", "127.0.0.1", "localhost"):
        m = ZENODO_RECORD_RE.fullmatch(parsed.path)
        if m:
            api_url = f"{parsed.scheme}://{parsed.netloc}/api/records/{m.group('id')}"
    elif host in ("doi.org", "dx.doi.org"):
        m = ZENODO_DOI_RE.fullmatch(parsed.path)
        if m:
            api_url = f"https://zenodo.org/api/records/{m.group('id')}"

    if api_url is None:
        return url, None

    req = urllib.request.Request(
        api_url,
        headers={"Accept": "application/json", "User-Agent": "wrfcomp-fetch-scoring-data/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(DEFAULT_MAX_METADATA_BYTES + 1)
        if len(raw) > DEFAULT_MAX_METADATA_BYTES:
            raise FetchError("Zenodo record metadata exceeds maximum allowed size")
        meta = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise FetchError(f"Failed to query Zenodo record metadata at {api_url}: {exc}") from exc

    files = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(files, list) or not files:
        raise FetchError(f"Zenodo record at {api_url} lists no downloadable files")

    target_prefix = f"wrfcomp-scoring-{year}.tar"
    candidates = [
        f
        for f in files
        if isinstance(f, dict)
        and str(f.get("key", "")).startswith(target_prefix)
        and not str(f.get("key", "")).endswith(".sha256")
    ]
    if not candidates:
        keys = [str(f.get("key", "")) for f in files if isinstance(f, dict)]
        raise FetchError(f"Zenodo record has no {target_prefix}* archive (found: {', '.join(keys)})")
    chosen = sorted(candidates, key=lambda f: str(f["key"]))[0]
    links = chosen.get("links") if isinstance(chosen.get("links"), dict) else {}
    file_url = links.get("self") or links.get("content")
    if not isinstance(file_url, str) or not file_url:
        raise FetchError(f"Zenodo file entry {chosen.get('key')!r} has no download link")
    _check_url_scheme(file_url)
    checksum = chosen.get("checksum")
    expected_md5: str | None = None
    if isinstance(checksum, str) and checksum.startswith("md5:"):
        expected_md5 = checksum.split(":", 1)[1].strip().lower()
    return file_url, expected_md5


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: float = 300.0,
    max_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
) -> int:
    """Download ``url`` to ``dest`` while enforcing a hard streaming byte ceiling."""
    _check_url_scheme(url)
    req = urllib.request.Request(url, headers={"User-Agent": "wrfcomp-fetch-scoring-data/1"})
    written = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise FetchError(
                        f"Download exceeded maximum allowed archive size ({max_bytes} bytes)"
                    )
                fh.write(chunk)
    except FetchError:
        dest.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        dest.unlink(missing_ok=True)
        raise FetchError(f"Download failed for {url}: {exc}") from exc
    return written


def _safe_extract_archive(
    archive_path: Path,
    staging_dir: Path,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_tar_headers: int | None = None,
    max_tar_metadata_bytes: int = DEFAULT_MAX_TAR_METADATA_BYTES,
    max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
    max_readme_bytes: int = DEFAULT_MAX_README_BYTES,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> set[str]:
    """Stream-extract plain USTAR regular files/directories with strict stream and metadata budgets."""
    extracted_files: set[str] = set()
    staging_real = staging_dir.resolve(strict=True)
    member_count = 0
    total_written = 0
    header_limit = (
        min(DEFAULT_MAX_TAR_HEADERS, max_members)
        if max_tar_headers is None
        else max_tar_headers
    )

    try:
        fh, _ = _open_regular_nofollow(archive_path, f"archive {archive_path.name}")
        with fh:
            magic = fh.read(2)
            fh.seek(0)
            decompressed: Any
            if magic == b"\x1f\x8b":
                decompressed = gzip.GzipFile(fileobj=fh, mode="rb")
            else:
                decompressed = fh
            try:
                bounded = _TarBudgetStream(
                    decompressed,
                    max_headers=header_limit,
                    max_metadata_bytes=max_tar_metadata_bytes,
                    max_member_bytes=max_member_bytes,
                    max_total_bytes=max_total_bytes,
                )
                with tarfile.open(fileobj=bounded, mode="r|") as tf:
                    for member in tf:
                        member_count += 1
                        if member_count > max_members:
                            raise FetchError(
                                f"Archive exceeds maximum allowed member count ({max_members})"
                            )
                        if member.pax_headers:
                            raise FetchError(
                                f"Refusing tar member with PAX extended headers: {member.name!r}"
                            )
                        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                            raise FetchError(f"Refusing non-regular tar member: {member.name!r}")
                        if not (member.isfile() or member.isdir()) or member.type not in (
                            tarfile.REGTYPE,
                            tarfile.AREGTYPE,
                            tarfile.DIRTYPE,
                        ):
                            raise FetchError(f"Refusing unsupported tar member type: {member.name!r}")
                        clean_name = member.name.rstrip("/")
                        if not clean_name or clean_name == ".":
                            continue
                        rel = _validate_relpath(clean_name, f"archive member {member.name!r}")
                        rel_posix = rel.as_posix()
                        dest = (staging_dir / rel).resolve()
                        if not dest.is_relative_to(staging_real):
                            raise FetchError(
                                f"Archive member escapes staging directory: {member.name!r}"
                            )
                        if member.isdir():
                            dest.mkdir(parents=True, exist_ok=True)
                            continue
                        if rel_posix in extracted_files:
                            raise FetchError(f"Duplicate archive member: {rel_posix!r}")
                        if member.size < 0:
                            raise FetchError(f"Negative member size in archive: {rel_posix!r}")

                        per_limit = max_member_bytes
                        if rel_posix == "MANIFEST.json":
                            per_limit = min(per_limit, max_manifest_bytes)
                        elif rel_posix == "README.md":
                            per_limit = min(per_limit, max_readme_bytes)
                        if member.size > per_limit:
                            raise FetchError(
                                f"Archive member {rel_posix!r} size {member.size} exceeds limit {per_limit}"
                            )
                        if total_written + member.size > max_total_bytes:
                            raise FetchError(
                                f"Archive expanded size exceeds limit ({max_total_bytes} bytes)"
                            )
                        free_bytes = shutil.disk_usage(staging_real).free
                        if free_bytes - member.size < min_free_bytes:
                            raise FetchError(
                                f"Insufficient staging disk space for {rel_posix!r} "
                                f"({free_bytes} bytes free)"
                            )

                        dest.parent.mkdir(parents=True, exist_ok=True)
                        src = tf.extractfile(member)
                        if src is None:
                            raise FetchError(f"Cannot read archive member: {member.name!r}")
                        member_written = 0
                        with src, dest.open("wb") as out_fh:
                            while True:
                                chunk = src.read(1024 * 1024)
                                if not chunk:
                                    break
                                member_written += len(chunk)
                                total_written += len(chunk)
                                if member_written > per_limit:
                                    raise FetchError(
                                        f"Archive member {rel_posix!r} stream exceeded limit {per_limit}"
                                    )
                                if total_written > max_total_bytes:
                                    raise FetchError(
                                        f"Archive expanded stream exceeded limit ({max_total_bytes} bytes)"
                                    )
                                out_fh.write(chunk)
                        if member_written != member.size:
                            raise FetchError(
                                f"Truncated or mismatched stream for {rel_posix!r}: "
                                f"wrote {member_written}, header {member.size}"
                            )
                        extracted_files.add(rel_posix)
            finally:
                if decompressed is not fh:
                    decompressed.close()
    except FetchError:
        raise
    except (tarfile.TarError, gzip.BadGzipFile, EOFError, OSError) as exc:
        raise FetchError(f"Failed to read archive {archive_path.name}: {exc}") from exc
    return extracted_files


def verify_staged_bundle(
    staging_dir: Path,
    extracted_files: set[str],
    *,
    year: int,
    references_path: Path,
    cfg: HarnessConfig,
    expected_manifest_bytes: bytes | None = None,
    max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
) -> dict[str, Any]:
    """Verify staged archive contents against ``MANIFEST.json``, strict layout, and ``REFERENCES.json``."""
    manifest_path = staging_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FetchError("Archive is missing MANIFEST.json")
    manifest_raw = _read_bounded_bytes(manifest_path, max_manifest_bytes, "MANIFEST.json")
    if expected_manifest_bytes is not None and manifest_raw != expected_manifest_bytes:
        raise FetchError("Archive MANIFEST.json does not match trusted --expected-manifest")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FetchError(f"Cannot parse MANIFEST.json: {exc}") from exc

    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise FetchError(f"Unsupported manifest_version: {manifest.get('manifest_version')!r}")
    if manifest.get("year") != year:
        raise FetchError(f"Manifest year {manifest.get('year')!r} does not match requested year {year}")

    actual_ref_sha = _sha256_file(references_path)
    if manifest.get("references_sha256") != actual_ref_sha:
        raise FetchError(
            f"MANIFEST.json references_sha256 ({manifest.get('references_sha256')}) "
            f"does not match {references_path.name} ({actual_ref_sha})"
        )

    expected_periods, pins = expected_pins(references_path, year, cfg)
    if manifest.get("periods") != expected_periods:
        raise FetchError(
            f"Manifest periods {manifest.get('periods')!r} do not match REFERENCES.json {expected_periods!r}"
        )

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise FetchError("MANIFEST.json has no entries")

    manifest_paths: dict[str, dict[str, Any]] = {}
    allowed_files: set[str] = {"MANIFEST.json", "README.md"}

    for entry in entries:
        if not isinstance(entry, dict):
            raise FetchError("Invalid entry in MANIFEST.json")
        relpath = entry.get("path")
        _validate_relpath(str(relpath), "MANIFEST.json entry")
        if relpath in manifest_paths:
            raise FetchError(f"Duplicate entry in MANIFEST.json: {relpath}")
        manifest_paths[relpath] = entry
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise FetchError(f"MANIFEST.json entry {relpath} has no files list")
        entry_total = 0
        for fmeta in files:
            fpath = fmeta.get("path")
            _validate_relpath(str(fpath), f"MANIFEST.json file under {relpath}")
            if fpath != relpath and not fpath.startswith(f"{relpath}/"):
                raise FetchError(f"MANIFEST.json file {fpath!r} does not lie under {relpath!r}")
            staged_file = _assert_tree_no_symlinks(
                staging_dir, str(fpath), f"staged file {fpath}", must_exist=True
            )
            if not stat.S_ISREG(os.lstat(staged_file).st_mode):
                raise FetchError(f"Missing regular file listed in MANIFEST.json: {fpath}")
            actual_size = os.lstat(staged_file).st_size
            if actual_size != fmeta.get("size_bytes"):
                raise FetchError(
                    f"Size mismatch for {fpath}: got {actual_size}, manifest has {fmeta.get('size_bytes')}"
                )
            actual_sha = _sha256_file(staged_file)
            if actual_sha != fmeta.get("sha256"):
                raise FetchError(
                    f"Byte SHA-256 mismatch for {fpath}: got {actual_sha}, manifest has {fmeta.get('sha256')}"
                )
            entry_total += actual_size
            allowed_files.add(str(fpath))
        if entry_total != entry.get("size_bytes"):
            raise FetchError(
                f"Entry total size mismatch for {relpath}: sum={entry_total}, "
                f"manifest={entry.get('size_bytes')}"
            )

    unexpected = sorted(extracted_files - allowed_files)
    if unexpected:
        raise FetchError(f"Archive contains unmanifested extra files: {unexpected[:5]}")

    missing_from_manifest = sorted(set(pins) - set(manifest_paths))
    extra_in_manifest = sorted(set(manifest_paths) - set(pins))
    if missing_from_manifest:
        raise FetchError(f"Archive is missing pinned artifacts: {missing_from_manifest}")
    if extra_in_manifest:
        raise FetchError(f"Archive contains unpinned artifacts: {extra_in_manifest}")

    # Strictly validate physical Zarr v2 / NetCDF layout + approved metadata values + schema
    # + chunk bounds BEFORE hashing, then verify dataset content_sha256 against MANIFEST & REFERENCES.
    for relpath, (kind, ref_sha256, expected_model_id) in sorted(pins.items()):
        entry = manifest_paths[relpath]
        if entry.get("kind") != kind:
            raise FetchError(f"Kind mismatch for {relpath}: manifest={entry.get('kind')}, expected={kind}")
        if entry.get("content_sha256") != ref_sha256:
            raise FetchError(
                f"MANIFEST.json content_sha256 disagrees with REFERENCES.json for {relpath}: "
                f"manifest={entry.get('content_sha256')}, pinned={ref_sha256}"
            )
        staged_target = _assert_tree_no_symlinks(
            staging_dir, relpath, f"staged {kind} {relpath}", must_exist=True
        )
        fmt = entry.get("format")
        if fmt == "zarr":
            expected_var_names = _validate_zarr_store(
                staging_dir,
                relpath,
                cfg=cfg,
                expected_model_id=expected_model_id,
                max_chunk_bytes=max_chunk_bytes,
                max_dataset_bytes=max_dataset_bytes,
            )
        elif fmt == "netcdf":
            expected_var_names = _validate_netcdf_file(staging_dir, relpath, kind=kind, cfg=cfg)
        else:
            raise FetchError(f"Unsupported format {fmt!r} for {relpath}")

        with _open_dataset(staged_target, fmt, f"{kind} {relpath}") as ds:
            _check_dataset_bounds(ds, relpath, max_dataset_bytes=max_dataset_bytes)
            _check_not_sealed(ds, kind, relpath, cfg)
            try:
                if kind == "truth":
                    validate_truth_dataset(ds, cfg=cfg)
                elif kind == "thresholds":
                    validate_thresholds(ds)
                elif kind == "reference_forecast":
                    validate_forecast_dataset(ds, cfg=cfg)
            except SchemaError as exc:
                raise FetchError(f"{relpath}: harness schema validation failed: {exc}") from exc
            ds_vars = {str(name) for name in ds.variables}
            if ds_vars != expected_var_names:
                raise FetchError(
                    f"{relpath}: container datasets {sorted(expected_var_names)} "
                    f"do not match xarray variables {sorted(ds_vars)}"
                )
            actual_content = content_sha256(ds)
        if actual_content != ref_sha256:
            raise FetchError(
                f"Dataset content_sha256 mismatch for {relpath}: "
                f"computed={actual_content}, pinned={ref_sha256}"
            )

    return manifest


def _mkdir_contained_no_symlinks(root: Path, rel_parent: Path, what: str) -> None:
    """Create parent directories under ``root`` while refusing any symlinked ancestor."""
    root_real = root.resolve(strict=True)
    cur = root
    for part in rel_parent.parts:
        cur = cur / part
        if os.path.lexists(cur):
            st = os.lstat(cur)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise FetchError(f"{what}: unsafe non-directory/symlink ancestor {cur.relative_to(root)}")
        else:
            cur.mkdir(mode=0o755)
        if not cur.resolve().is_relative_to(root_real):
            raise FetchError(f"{what}: parent escapes destination root")


def _verify_existing_destination_matches_manifest(
    data_dir: Path,
    dest: Path,
    entry: dict[str, Any],
) -> None:
    """Require an existing destination to match the manifest's physical file inventory and byte hashes."""
    relpath = entry["path"]
    fmt = entry["format"]
    expected_files: dict[str, tuple[int, str]] = {
        str(f["path"]): (int(f["size_bytes"]), str(f["sha256"])) for f in entry["files"]
    }
    st = os.lstat(dest)
    if fmt == "netcdf":
        if not stat.S_ISREG(st.st_mode):
            raise FetchError(
                f"Destination {relpath} already exists and is not a regular file; "
                "re-run with --overwrite to replace it"
            )
        exp_size, exp_sha = expected_files[relpath]
        if st.st_size != exp_size:
            raise FetchError(
                f"Destination {relpath} already exists with different bytes "
                f"(size {st.st_size} != manifest {exp_size}); re-run with --overwrite to replace it"
            )
        actual_sha = _sha256_file(dest)
        if actual_sha != exp_sha:
            raise FetchError(
                f"Destination {relpath} already exists with different bytes "
                f"(byte SHA-256 mismatch vs manifest: found {actual_sha}, expected {exp_sha}); "
                "re-run with --overwrite to replace it"
            )
        return

    if fmt == "zarr":
        if not stat.S_ISDIR(st.st_mode):
            raise FetchError(
                f"Destination {relpath} already exists and is not a directory; "
                "re-run with --overwrite to replace it"
            )
        existing_files: set[str] = set()
        for dirpath, _dirnames, filenames in os.walk(dest, followlinks=False):
            dp = Path(dirpath)
            for fname in filenames:
                fpath = dp / fname
                existing_files.add(fpath.relative_to(data_dir).as_posix())
        if existing_files != set(expected_files):
            missing = sorted(set(expected_files) - existing_files)
            extra = sorted(existing_files - set(expected_files))
            raise FetchError(
                f"Destination {relpath} already exists with different file inventory "
                f"(missing={missing[:3]}, extra={extra[:3]}); re-run with --overwrite to replace it"
            )
        for frel, (exp_size, exp_sha) in sorted(expected_files.items()):
            fpath = data_dir / frel
            fst = os.lstat(fpath)
            if fst.st_size != exp_size:
                raise FetchError(
                    f"Destination {frel} already exists with different bytes "
                    f"(size {fst.st_size} != manifest {exp_size}); re-run with --overwrite to replace it"
                )
            actual_sha = _sha256_file(fpath)
            if actual_sha != exp_sha:
                raise FetchError(
                    f"Destination {frel} already exists with different bytes "
                    f"(byte SHA-256 mismatch vs manifest: found {actual_sha}, expected {exp_sha}); "
                    "re-run with --overwrite to replace it"
                )
        return

    raise FetchError(f"Unsupported format {fmt!r} for {relpath}")


def _publish_staged(
    staging_dir: Path,
    data_dir: Path,
    entries: list[dict[str, Any]],
    *,
    overwrite: bool,
    cfg: HarnessConfig,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
) -> list[str]:
    """Atomically install verified entries from ``staging_dir`` into ``data_dir``."""
    actions: list[tuple[str, str]] = []
    for entry in entries:
        relpath = entry["path"]
        dest = _assert_tree_no_symlinks(
            data_dir, relpath, f"destination {relpath}", must_exist=False
        )
        if not os.path.lexists(dest):
            actions.append((relpath, "install"))
            continue
        if overwrite:
            actions.append((relpath, "replace"))
            continue
        # Keep an existing destination ONLY when its physical file inventory and byte hashes match
        # the verified manifest byte for byte (never on logical hash equivalence alone).
        _verify_existing_destination_matches_manifest(data_dir, dest, entry)
        actions.append((relpath, "keep"))

    installed: list[str] = []
    for relpath, action in actions:
        if action == "keep":
            continue
        rel = _validate_relpath(relpath, f"install {relpath}")
        if rel.parent != Path("."):
            _mkdir_contained_no_symlinks(data_dir, rel.parent, f"destination {relpath}")
        dest = _assert_tree_no_symlinks(
            data_dir, relpath, f"destination {relpath}", must_exist=(action == "replace")
        )
        src = staging_dir / rel
        if action == "replace":
            backup = dest.parent / f".{dest.name}.old.{os.getpid()}"
            if os.path.lexists(backup):
                if stat.S_ISDIR(os.lstat(backup).st_mode):
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
            os.replace(dest, backup)
            try:
                os.replace(src, dest)
            except Exception:
                os.replace(backup, dest)
                raise
            if stat.S_ISDIR(os.lstat(backup).st_mode):
                shutil.rmtree(backup)
            else:
                backup.unlink()
        else:
            os.replace(src, dest)
        installed.append(relpath)
    return installed


def _parse_sha256_file(path: Path) -> str:
    raw = _read_bounded_bytes(path, 4096, f"sha256 file {path.name}").decode("utf-8").strip()
    if not raw:
        raise FetchError(f"Empty sha256 file: {path.name}")
    token = raw.splitlines()[0].split()[0].strip().lower()
    if not SHA256_RE.fullmatch(token):
        raise FetchError(f"Invalid SHA-256 digest in {path.name}: {token!r}")
    return token


def fetch_scoring_data(
    *,
    year: int,
    data_dir: Path,
    url: str | None = None,
    archive: Path | None = None,
    references_path: Path = DEFAULT_REFERENCES,
    config_path: Path = DEFAULT_CONFIG,
    expected_sha256: str | None = None,
    expected_sha256_file: Path | None = None,
    expected_manifest: Path | None = None,
    allow_unverified_archive: bool = False,
    overwrite: bool = False,
    keep_archive: bool = False,
    timeout: float = 300.0,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_tar_headers: int | None = None,
    max_tar_metadata_bytes: int = DEFAULT_MAX_TAR_METADATA_BYTES,
    max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
) -> dict[str, Any]:
    """Download (or open local), verify, and unpack a scoring data bundle into ``data_dir``."""
    if (url is None) == (archive is None):
        raise FetchError("Specify exactly one of --url or --archive")

    resolved_sha256: str | None = None
    if expected_sha256 is not None:
        token = expected_sha256.strip().lower()
        if not SHA256_RE.fullmatch(token):
            raise FetchError(f"Invalid --expected-sha256 digest: {expected_sha256!r}")
        resolved_sha256 = token
    if expected_sha256_file is not None:
        file_sha = _parse_sha256_file(expected_sha256_file)
        if resolved_sha256 is not None and resolved_sha256 != file_sha:
            raise FetchError("--expected-sha256 disagrees with --expected-sha256-file")
        resolved_sha256 = file_sha

    expected_manifest_bytes: bytes | None = None
    if expected_manifest is not None:
        expected_manifest_bytes = _read_bounded_bytes(
            expected_manifest, max_manifest_bytes, f"expected manifest {expected_manifest.name}"
        )

    if resolved_sha256 is None and expected_manifest_bytes is None and not allow_unverified_archive:
        raise FetchError(
            "An independently trusted physical pin is required: pass --expected-sha256, "
            "--expected-sha256-file, or --expected-manifest (or pass --allow-unverified-archive "
            "to skip archive-level pinning)"
        )

    cfg = load_config(config_path)
    expected_pins(references_path, year, cfg)

    if os.path.lexists(data_dir) and stat.S_ISLNK(os.lstat(data_dir).st_mode):
        raise FetchError("Destination --data-dir cannot be a symlink")
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".fetch-scoring-", dir=data_dir) as tmp_str:
        tmp_dir = Path(tmp_str)
        staging_dir = tmp_dir / "stage"
        staging_dir.mkdir()

        expected_md5: str | None = None
        if url is not None:
            file_url, expected_md5 = resolve_download_url(url, year, timeout=min(timeout, 60.0))
            archive_path = tmp_dir / f"wrfcomp-scoring-{year}.tar.gz"
            download_file(file_url, archive_path, timeout=timeout, max_bytes=max_archive_bytes)
        else:
            assert archive is not None
            if os.path.lexists(archive) and stat.S_ISLNK(os.lstat(archive).st_mode):
                raise FetchError(f"Refusing symlinked --archive: {archive.name}")
            archive_path = archive.resolve()
            if not archive_path.is_file():
                raise FetchError(f"Archive not found: {archive_path.name}")
            if archive_path.stat().st_size > max_archive_bytes:
                raise FetchError(
                    f"Archive size exceeds maximum allowed ({max_archive_bytes} bytes)"
                )

        archive_sha256 = _sha256_file(archive_path)
        if resolved_sha256 is not None and archive_sha256 != resolved_sha256:
            raise FetchError(
                f"Archive SHA-256 mismatch: got {archive_sha256}, expected {resolved_sha256}"
            )
        if expected_md5 is not None:
            actual_md5 = _md5_file(archive_path)
            if actual_md5 != expected_md5:
                raise FetchError(
                    f"Archive MD5 mismatch vs Zenodo record metadata: "
                    f"got {actual_md5}, expected {expected_md5}"
                )

        extracted_files = _safe_extract_archive(
            archive_path,
            staging_dir,
            max_members=max_members,
            max_member_bytes=max_member_bytes,
            max_total_bytes=max_total_bytes,
            max_tar_headers=max_tar_headers,
            max_tar_metadata_bytes=max_tar_metadata_bytes,
            max_manifest_bytes=max_manifest_bytes,
        )
        manifest = verify_staged_bundle(
            staging_dir,
            extracted_files,
            year=year,
            references_path=references_path,
            cfg=cfg,
            expected_manifest_bytes=expected_manifest_bytes,
            max_manifest_bytes=max_manifest_bytes,
            max_chunk_bytes=max_chunk_bytes,
            max_dataset_bytes=max_dataset_bytes,
        )
        installed = _publish_staged(
            staging_dir,
            data_dir,
            manifest["entries"],
            overwrite=overwrite,
            cfg=cfg,
            max_chunk_bytes=max_chunk_bytes,
            max_dataset_bytes=max_dataset_bytes,
        )

        if keep_archive and url is not None:
            _mkdir_contained_no_symlinks(data_dir, Path("bundle"), "bundle archive copy")
            bundle_dest = data_dir / "bundle" / f"wrfcomp-scoring-{year}.tar.gz"
            if os.path.lexists(bundle_dest) and stat.S_ISLNK(os.lstat(bundle_dest).st_mode):
                raise FetchError("Refusing symlinked destination for --keep-archive")
            shutil.copy2(archive_path, bundle_dest)

    return {
        "archive_sha256": archive_sha256,
        "entries_verified": len(manifest["entries"]),
        "installed": installed,
        "periods": manifest["periods"],
        "year": year,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download, verify, and unpack a WRFComp scoring data bundle."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--url",
        type=str,
        default=None,
        help="Zenodo record URL (or direct archive URL) for the scoring bundle.",
    )
    source.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Local bundle archive path (offline mode; no network access).",
    )
    parser.add_argument("--year", type=int, default=2024, help="Scoring bundle year (default: 2024).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Target WRFComp data directory (default: $WRFCOMP_DATA_DIR or ~/wrfcomp-data).",
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
        "--expected-sha256",
        type=str,
        default=None,
        help="Trusted SHA-256 hex digest of the archive (required unless --expected-manifest is set).",
    )
    parser.add_argument(
        "--expected-sha256-file",
        type=Path,
        default=None,
        help="Path to a .sha256 sidecar file containing the trusted archive SHA-256.",
    )
    parser.add_argument(
        "--expected-manifest",
        type=Path,
        default=None,
        help="Path to an independently trusted MANIFEST.json sidecar that the archive must match.",
    )
    parser.add_argument(
        "--allow-unverified-archive",
        action="store_true",
        help="Allow installing without --expected-sha256 or --expected-manifest (not recommended).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files/stores in --data-dir even if their hashes differ.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep a copy of the downloaded archive under <data-dir>/bundle/.",
    )
    args = parser.parse_args(argv)
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    try:
        summary = fetch_scoring_data(
            year=args.year,
            data_dir=data_dir,
            url=args.url,
            archive=args.archive,
            references_path=args.references,
            config_path=args.config,
            expected_sha256=args.expected_sha256,
            expected_sha256_file=args.expected_sha256_file,
            expected_manifest=args.expected_manifest,
            allow_unverified_archive=args.allow_unverified_archive,
            overwrite=args.overwrite,
            keep_archive=args.keep_archive,
        )
    except FetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
