# Station forecast baselines

Run from the checkout with `uv sync --extra dev --extra grib` (the `grib` extra is
required for `gefs`; `climatology` and `persistence` only need `uv sync --extra dev`).
The following command builds one initialization across all configured stations. Outputs go under
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

Climatology and persistence write `model_version="3"` and cover leads of 1, 3, 6, 12,
24 and 48 hours. Both use observations from 2014 onward for calibration. The inclusive
cutoff is the earlier of the configured training end and the earliest initialization in
the request. A report's measurement time is its hourly label plus its variable-specific
`X_offset_min`, which must lie within ±15 minutes.

For a request before the training end, that default cutoff follows the request. A confirmation period
inside the training years passes a fixed cutoff instead (`train_end` in `forecast`, recorded in the
store's `train_end_utc` attribute); see `docs/CONFIRMATION_2023.md`.

Both models use the same member rule. For `M` members, member `i` uses the empirical
quantile at `(i + 0.5) / M`, for `i = 0, …, M − 1`. This selects the smallest sampled
value whose cumulative frequency reaches that probability (NumPy's `inverted_cdf`).
Values can repeat. The default ensemble size is the larger of 30 and
`min_members_for_gate` in `configs/splits.yaml` (currently 20). Explicit member counts
must meet that floor.

**Climatology** builds a sample for each station, variable and forecast valid time.
Members come from the first pool with a finite observation whose measurement time is
at or before the calibration cutoff. The search order is:

1. The target UTC hour within seven calendar days of the target date.
2. The same UTC hour within 30 calendar days.
3. Any UTC hour in the target calendar month.

Both day windows wrap across New Year on a common leap-year calendar. Each pool uses
the same station and variable. If all three pools are empty, every member is NaN.

**Persistence** checks the hourly slots from initialization back through three hours.
For each station and variable, it takes the latest slot with a finite observation
measured by initialization. Historical starting values use the same rule. For each
station, variable, lead and UTC initialization hour, the calibration sample contains
the verifying observation minus the historical starting value for each eligible pair.
Both ends come from training data, and the verifying report must have been measured
by the cutoff. Each forecast member adds one empirical error quantile to the current
starting value. If the three-hour lookback yields no eligible starting value, or if the
error sample is empty, persistence uses the climatology members for that station,
variable and valid time. If climatology also exhausts its pools, the point remains NaN.

## NOAA GEFS

Read the NOAA public S3 bucket using unsigned HTTPS range requests. The half-degree
`atmos/pgrb2ap5` A files cover c00 and p01 through p30 at 3, 6, 12, 24 and 48 hours.
Index offsets delimit individual GRIB2 messages. The decoder uses `eccodes` and the
native `eccodeslib` wheel on CPU via the optional `grib` extra (`uv sync --extra grib`;
Linux, macOS 15+, or WSL2 on Windows). The table follows the archived 2024-01-01 00Z inventories.
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

### Known limits on real data (January and July 2024)

These are measured properties of the raw GEFS baseline at 500 vetted stations, using 62
initializations per month (00Z and 12Z) and hourly ISD truth. **No lapse rate was tuned
on validation data.** The comparison uses the same ensemble members and observed
station-hours in both columns. “Raw” means bilinearly interpolated 2 m temperature
before the height adjustment. “Corrected” includes the protocol's fixed 6.5 K/km
adjustment. Bias is ensemble-mean forecast minus observation.

| 24-hour t2m diagnostic | January raw | January corrected | July raw | July corrected |
| --- | ---: | ---: | ---: | ---: |
| Gate 1 skill over climatology (fair composite loss) | 51.40% | 54.70% | −3.01% | +3.68% |
| Ensemble-mean RMSE (K) | 2.71 | 2.61 | 2.64 | 2.52 |
| Station mean bias slope versus station minus model elevation (K/km) | +4.96 | −1.54 | +4.21 | −2.29 |

The negative residual slopes show that the fixed adjustment was stronger than the
observed cross-station gradient in both months, even as it improved skill. July's
+0.41 correlation (+0.09 in January) is between corrected station bias and **station
elevation alone**. The correction acts on the **difference** between station and model
elevation. Across all five leads, July skill averaged −2.54% raw and +3.76% corrected.
[January scorecard](../experiments/gefs_2024-01.json) · [July scorecard](../experiments/gefs_2024-07.json)

At 24 hours in July, 13 stations had absolute mean bias above 3 K (ten warm, three
cold). The half-degree grid places terrain hundreds of metres above three coastal
airports and close to sea level at Catalina. In each case the adjustment amplified
an existing error.

| Airport | Station / model elevation (m) | Applied adjustment (K) | Raw bias (K) | Corrected bias (K) |
| --- | ---: | ---: | ---: | ---: |
| Santa Barbara Municipal | 2.5 / 363.4 | +2.35 | +4.67 | +7.02 |
| Oxnard | 11.9 / 356.1 | +2.24 | +3.77 | +6.00 |
| Salinas Municipal | 22.3 / 332.9 | +2.02 | +3.77 | +5.79 |
| Catalina | 477.0 / 4.6 | −3.07 | −2.52 | −5.59 |

Across all five July leads, daytime RMSE fell from 2.37 K raw to 2.32 K corrected,
and nighttime RMSE fell from 2.97 K to 2.71 K. Day and night use solar elevation at
the station and forecast valid time (above 10° and below 0°, respectively), excluding
twilight. Several inland airports still had a large nighttime warm bias. At 24 hours,
corrected day and night biases were −0.21 and +7.18 K at Grand Canyon National Park,
+0.06 and +6.55 K at Bryce Canyon, and −0.45 and +6.86 K at Derby Field. The height
adjustment is constant at each station, so it cannot account for the day–night swing.

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
Bulk downloads require a maintainer's go.
