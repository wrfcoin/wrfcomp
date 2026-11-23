"""Immutable, aligned observation columns. Array contents are omitted from repr."""

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True, repr=False)
class ObservationBatch:
    station_ids: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    elevation_m: np.ndarray
    obs_time: np.ndarray
    sensor_type_id: np.ndarray
    variable: np.ndarray
    values: np.ndarray
    qc_flags: np.ndarray | None = None

    def __post_init__(self):
        size = len(self.station_ids)
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "qc_flags" and value is None:
                value = np.zeros(size, dtype=np.uint32)
            dtype = {"qc_flags": np.uint32, "obs_time": "datetime64[ns]", "values": np.float32,
                     "station_ids": str, "variable": str, "lat": float, "lon": float,
                     "elevation_m": float}.get(field.name)
            column = np.array(value, dtype=dtype, copy=True)
            if column.shape != (size,):
                raise ValueError("observation columns must be aligned one-dimensional arrays")
            column.flags.writeable = False
            object.__setattr__(self, field.name, column)
        if np.any(np.char.str_len(self.station_ids) != 11) or np.any(np.isnat(self.obs_time)):
            raise ValueError("invalid station identifier or timestamp")
        if not (np.all(np.abs(self.lat) <= 90) and np.all(np.abs(self.lon) <= 180)
                and np.all(np.isfinite(self.elevation_m))):
            raise ValueError("invalid station location")
