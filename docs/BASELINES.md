# Station forecast baselines

Run from the checkout with `uv sync --extra dev`. The following command builds one
initialization across all configured stations. Outputs go under
`$WRFCOMP_DATA_DIR/baselines` (default `~/wrfcomp-data/baselines`).

```sh
uv run python scripts/build_baselines.py gefs 2024-01-01T00
```

For `climatology` or `persistence`, supply `--truth /absolute/path/to/truth.zarr`
or a NetCDF file under the data directory. An archive for persistence must include the
training years and the recent hourly slots at initialization. Station IDs retain leading
zeros. Output has one initialization, float32 station values, integer member IDs, and
UTC times. A run refuses windows that would reach the sealed period.

## Historical baselines

Both baselines use 2014 through the inclusive training end in `configs/splits.yaml`.
Climatology selects finite observations at the station and UTC valid hour within seven
calendar days of the target on a circular leap-year calendar. Per variable it takes
30 evenly spaced ranks in chronological order, rounding to the nearest index. Short
samples repeat members. An empty sample yields NaN.

Persistence checks hourly slots from initialization back through three hours, independently
for each station and variable. The first finite observation with a valid offset and
`valid_time + X_offset_min <= init_time` supplies the starting value. Without an eligible
slot the forecast is NaN and the point is excluded for persistence only. At historical
initializations with the same UTC hour, compute the verifying value minus the historical
starting value for the same station, variable and lead. Both ends use training data. Add
the linearly interpolated error quantiles at probabilities `(i + 0.5) / 30`,
`i = 0..29`, to the current starting value. Missing calibration pairs drop out. An empty
calibration sample yields NaN. Both models cover 1, 3, 6, 12, 24 and 48 hours.

## NOAA GEFS

Read the NOAA public S3 bucket using unsigned HTTPS range requests. The half-degree
`atmos/pgrb2ap5` A files cover c00 and p01 through p30 at 3, 6, 12, 24 and 48 hours.
Index offsets delimit individual GRIB2 messages. The decoder uses `eccodes` and the
native `eccodeslib` wheel on CPU. The table follows the archived 2024-01-01 00Z inventories.
[Control f003 index](https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240101/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f003.idx)
and [control analysis index](https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240101/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f000.idx).

| Output | Index field and level | Processing at station |
| --- | --- | --- |
| t2m (K) | TMP, 2 m above ground | `T_station = T_grid - 0.0065 * (z_station - z_model)`, with elevations in metres. |
| td2m (K) | RH and TMP, 2 m above ground | Magnus dewpoint from bilinear RH (%) and unadjusted air temperature, capped at corrected t2m. |
| u10, v10 (m/s) | UGRD, VGRD, 10 m above ground | Bilinear eastward and northward components. |
| msl (Pa) | PRMSL, mean sea level | Bilinear pressure. |
| Model elevation (m) | HGT, surface | Bilinear control analysis orography reused across the cycle. |

At most eight workers issue one request each at a time. GRIB messages are decoded in memory.
The restart cache stores extracted station arrays per completed member and lead, keyed by
cycle, algorithm version and station metadata. Rerun the command after interruption.
A completed Zarr is published after schema checks. Existing output is protected.

## One-cycle sizing

A cold run for 2024-01-01 00Z covered 500 stations, 31 members and five leads.
All five variables had 77,500 finite values apiece and zero dewpoints exceeded t2m.
The process received 183,566,140 HTTP payload bytes (indexes and GRIB ranges),
took 70.903 s and peaked at 199,784 KiB RSS on Linux.
The Zarr occupied 1,108,724 apparent bytes including directory entries and the restart
cache occupied 1,573,968. GRIB payloads were discarded.

| Requested scenario | Cycles | Estimated HTTP payload | Estimated serial cycle time |
| --- | ---: | ---: | ---: |
| January plus July 2024, 00Z and 12Z | 124 | 22.76 GB | 2.44 h |
| Calendar 2024, four cycles per day | 1,464 | 268.74 GB | 28.83 h |

Decimal GB. Estimates scale the single observation linearly, including cold-cache work on
each cycle. Compression, network latency and retries vary. The calendar-year figure is
a sizing estimate only and includes cycles outside the allowed validation window.
Bulk downloads require the sprint manager's go.
