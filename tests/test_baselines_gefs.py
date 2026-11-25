import sys

import numpy as np
import pytest
import xarray as xr

from wrfcomp.baselines import gefs
from wrfcomp.baselines.io import validate

# Consecutive rows from gec00.t00z.pgrb2a.0p50.f003.idx, 2024-01-01.
SAMPLE_INDEX = """63:11605460:d=2024010100:TMP:2 m above ground:3 hour fcst:ENS=low-res ctl
64:11843542:d=2024010100:RH:2 m above ground:3 hour fcst:ENS=low-res ctl
65:12056792:d=2024010100:TMAX:2 m above ground:0-3 hour max fcst:ENS=low-res ctl"""


def test_range_selection():
    ranges = gefs.index_ranges(SAMPLE_INDEX)
    assert ranges == {gefs.FIELDS["t2m"]: (11605460, 11843541), gefs.FIELDS["rh"]: (11843542, 12056791)}
    assert gefs.index_ranges(SAMPLE_INDEX.splitlines()[0])[gefs.FIELDS["t2m"]] == (11605460, None)


def test_bilinear_and_lapse():
    grid = np.array([[280.0, 284.0], [288.0, 292.0]])
    result = gefs.bilinear(grid, np.array([0.25, -0.25]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(result, [285, 285])
    np.testing.assert_allclose(gefs.lapse(result, 500, np.array([1500, 0])), [278.5, 288.25])


@pytest.mark.parametrize("humidity", [90.0, 100.04158782958984])
def test_extract_and_resume(monkeypatch, tmp_path, humidity):
    fields = [gefs.HEIGHT, *gefs.FIELDS.values()]
    fixture = "\n".join(f"{i + 1}:{i}:d=2024010100:{v}:{level}:fcst" for i, (v, level) in enumerate(fields))

    class Offline:
        def get(self, url, interval=None):
            if url.endswith(".idx"):
                return fixture.encode()
            return [100.0, 280.0, humidity, -2.0, 1.0, 99000.0][interval[0]]

    monkeypatch.setattr(gefs, "points", lambda value, stations: np.array([value]))
    stations = xr.Dataset(
        coords={
            "station_id": ["00123456789"],
            "longitude": ("station_id", [-104.0]),
            "latitude": ("station_id", [40.0]),
            "elevation_m": ("station_id", [1100.0]),
        }
    )
    out = gefs.extract(stations, "2024-01-01", tmp_path, Offline())
    validate(out)
    assert out.t2m.shape == (1, 5, 1, 31)
    assert (out.t2m == 273.5).all() and (out.td2m == out.t2m).all()
    xr.testing.assert_identical(out, gefs.extract(stations, "2024-01-01", tmp_path, None))
    with pytest.raises(ValueError, match="sealed"):
        gefs.extract(stations, "2024-12-31", tmp_path, None)


def test_points_missing_grib_extra_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "eccodes", None)
    stations = xr.Dataset(
        coords={
            "station_id": ["00123456789"],
            "longitude": ("station_id", [-104.0]),
            "latitude": ("station_id", [40.0]),
            "elevation_m": ("station_id", [1100.0]),
        }
    )
    with pytest.raises(ImportError, match=r"grib"):
        gefs.points(b"", stations)


def test_points_decodes_grib_message():
    ec = pytest.importorskip("eccodes", reason="requires the optional 'grib' extra (eccodes)")
    grib = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        ni, nj = ec.codes_get(grib, "Ni"), ec.codes_get(grib, "Nj")
        ec.codes_set_values(grib, np.full(ni * nj, 285.0, dtype=np.float64))
        message = ec.codes_get_message(grib)
    finally:
        ec.codes_release(grib)

    stations = xr.Dataset(
        coords={
            "station_id": ["00123456789"],
            "longitude": ("station_id", [2.0]),
            "latitude": ("station_id", [58.0]),
            "elevation_m": ("station_id", [0.0]),
        }
    )
    np.testing.assert_allclose(gefs.points(message, stations), [285.0])
