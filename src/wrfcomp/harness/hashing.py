"""Content hash of an xarray dataset.

The hash covers, for every variable (data variables and coordinates alike) in sorted name order:
the name, whether it is a coordinate, its dimension names, its shape, a canonical dtype label and its
values in C order. It does not cover attributes, encodings, chunking, compression or file layout, so
the same data written twice with different chunks hashes the same.

Canonical forms (so that a store round trip cannot change the hash):

- numbers and booleans: little-endian bytes of the in-memory dtype; every NaN is written as the one
  canonical quiet NaN of that dtype;
- ``datetime64`` / ``timedelta64``: converted to nanosecond resolution, hashed as int64 (NaT included);
- strings (``str``, ``bytes``, object or numpy string dtypes): each element as UTF-8 with a length prefix.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import xarray as xr

HASH_FORMAT = b"wrfcomp-content-hash-v1"
_BLOCK_BYTES = 64 * 1024 * 1024


def _field(h: hashlib._Hash, data: bytes) -> None:
    """Length-prefixed write, so that neighbouring fields cannot run into each other."""
    h.update(struct.pack("<Q", len(data)))
    h.update(data)


def _dtype_label(dtype: np.dtype) -> str:
    kind = dtype.kind
    if kind == "M":
        return "datetime64[ns]"
    if kind == "m":
        return "timedelta64[ns]"
    if kind in "OUST":
        return "string"
    if kind in "biufc":
        return f"{kind}{dtype.itemsize}"
    raise TypeError(f"cannot hash dtype {dtype!r}")


def _values_bytes(values: np.ndarray, label: str) -> bytes:
    values = np.asarray(values)
    if label == "datetime64[ns]":
        return np.ascontiguousarray(values.astype("datetime64[ns]").astype("<i8")).tobytes()
    if label == "timedelta64[ns]":
        return np.ascontiguousarray(values.astype("timedelta64[ns]").astype("<i8")).tobytes()
    if label == "string":
        parts = []
        for item in values.ravel(order="C").tolist():
            if isinstance(item, bytes):
                raw = item
            elif isinstance(item, str):
                raw = item.encode("utf-8")
            else:
                raise TypeError(f"cannot hash non-string object {item!r}")
            parts.append(struct.pack("<Q", len(raw)))
            parts.append(raw)
        return b"".join(parts)
    out = values.astype(values.dtype.newbyteorder("<"), copy=True)
    if out.dtype.kind in "fc":
        out[np.isnan(out)] = np.nan
    return np.ascontiguousarray(out).tobytes()


def _update_variable(h: hashlib._Hash, name: str, var: xr.Variable, is_coord: bool) -> None:
    label = _dtype_label(var.dtype)
    _field(h, str(name).encode("utf-8"))
    _field(h, b"coord" if is_coord else b"data")
    _field(h, struct.pack("<Q", var.ndim))
    for dim, size in zip(var.dims, var.shape, strict=True):
        _field(h, str(dim).encode("utf-8"))
        _field(h, struct.pack("<Q", size))
    _field(h, label.encode("ascii"))
    if var.ndim == 0 or var.size == 0:
        _field(h, _values_bytes(var.values, label))
        return
    # Stream along the first axis: C-order bytes of consecutive slabs are the C-order bytes of the whole.
    row_bytes = max(1, (var.size // var.shape[0]) * max(var.dtype.itemsize, 8))
    step = max(1, _BLOCK_BYTES // row_bytes)
    h.update(struct.pack("<Q", var.size))
    for start in range(0, var.shape[0], step):
        h.update(_values_bytes(var[start : start + step].values, label))


def content_sha256(ds: xr.Dataset) -> str:
    """sha256 hex digest of the dataset's content. See the module docstring for what is covered."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError("content_sha256 expects an xarray.Dataset")
    h = hashlib.sha256()
    _field(h, HASH_FORMAT)
    names = sorted(str(name) for name in ds.variables)
    if len(set(names)) != len(names):
        raise ValueError("variable names are not unique after str()")
    by_name = {str(name): name for name in ds.variables}
    _field(h, struct.pack("<Q", len(names)))
    for name in names:
        key = by_name[name]
        _update_variable(h, name, ds.variables[key], key in ds.coords)
    return h.hexdigest()
