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
            dtype = {
                "qc_flags": np.uint32,
                "obs_time": "datetime64[ns]",
                "values": np.float32,
                "station_ids": str,
                "variable": str,
                "lat": float,
                "lon": float,
                "elevation_m": float,
            }.get(field.name)
            column = np.array(value, dtype=dtype, copy=True)
            if column.shape != (size,):
                raise ValueError("observation columns must be aligned one-dimensional arrays")
            column.flags.writeable = False
            object.__setattr__(self, field.name, column)
        if np.any(np.char.str_len(self.station_ids) != 11) or np.any(np.isnat(self.obs_time)):
            raise ValueError("invalid station identifier or timestamp")
        if not (
            np.all(np.abs(self.lat) <= 90)
            and np.all(np.abs(self.lon) <= 180)
            and np.all(np.isfinite(self.elevation_m))
        ):
            raise ValueError("invalid station location")


@dataclass(frozen=True)
class QCConfig:
    bounds: tuple = (
        ("t2m", 180, 340),
        ("td2m", 150, 340),
        ("u10", -120, 120),
        ("v10", -120, 120),
        ("wind_speed", 0, 150),
        ("wind_gust", 0, 180),
        ("station_pressure", 25000, 110000),
        ("msl", 85000, 110000),
        ("precip", 0, 1000),
    )
    steps: tuple = (
        ("t2m", 10),
        ("td2m", 10),
        ("u10", 40),
        ("v10", 40),
        ("msl", 2500),
        ("wind_speed", 40),
        ("wind_gust", 60),
        ("station_pressure", 2500),
        ("precip", 150),
    )
    step_minutes: int = 15
    flat_reports: int = 6
    sea_pressure_pa: float = 101325
    scale_height_m: float = 8434
    pressure_tolerance_pa: float = 10000
    radius_km: float = 25
    lapse_k_per_km: float = 6.5
    minimum_buddies: int = 3
    spatial_z: float = 4
    mad_scale: float = 1.4826
    scale_floor_fraction: float = 0.25
    day_degrees: float = 10
    huber_cut: float = 2.5
    gross_cut: float = 8
    corroboration_z: float = 3
    corroborating_stations: int = 2
    ema_alpha: float = 0.04
    solar_allowance: float = 1
    solar_scale: float = 2
    window_days: int = 7
    min_pairs: int = 12
    low_correlation: float = 0.5
    copy_correlation: float = 0.998
    copy_residual_sigma: float = 0.01
    guard_decay: float = 0.9
    official_prior: float = 1
    new_prior: float = 0.5
    calm_wind_ms: float = 0.5
    wind_flat_minutes: int = 75
    official_buddy_weight: int = 2
    verified_days: int = 7
    verified_reports_per_day: int = 6
    verified_reports: int = 84
    verified_bias_sigma: float = 1.0
    verified_prior: float = 1.0
