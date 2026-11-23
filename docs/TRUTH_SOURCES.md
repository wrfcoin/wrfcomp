# Ground-Truth Observation Sources and Parsing Rules

## NOAA ISD Status and Transition to GHCNh

NOAA NCEI has released the **Global Historical Climatology Network hourly (GHCNh)** as the
next-generation hourly and synoptic surface dataset that officially replaces the legacy
**Integrated Surface Database (ISD / "Global Hourly")**, alongside upgrading Local Climatological
Data from LCDv1 to LCDv2. According to NCEI's product pages and release announcement—fetched from
[https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database),
[https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly),
and
[https://www.ncei.noaa.gov/news/next-generation-climate-dataset-built-seamless-integration](https://www.ncei.noaa.gov/news/next-generation-climate-dataset-built-seamless-integration)—GHCNh
incorporates the ISD data sources plus more than 100 additional networks, vertically aligns station
identifiers with GHCN daily (GHCNd), and is distributed in PSV/Parquet formats. Legacy ISD station
metadata (`isd-history.csv`) stops at August 2025, reflecting the end of routine ISD updates.

Operationally, NCEI's ongoing cloud migration
([https://www.ncei.noaa.gov/news/cloud-migration](https://www.ncei.noaa.gov/news/cloud-migration))
has also retired direct HTTP CSV serving at the legacy NCEI folder path: requesting
`https://www.ncei.noaa.gov/data/global-hourly/access/2024/<station_id>.csv` on 2026-11-23 redirected
to the AWS S3 Explorer HTML page (`https://noaa-global-hourly-pds.s3.amazonaws.com/index.html`,
36,822 bytes of HTML) rather than returning CSV data. However, the official NOAA Open Data
Dissemination (NODD) S3 buckets remain directly readable with HTTP 200: global-hourly station-year
CSVs live at `https://noaa-global-hourly-pds.s3.amazonaws.com/<year>/<station_id>.csv` (verified on
2024 station files such as
[https://noaa-global-hourly-pds.s3.amazonaws.com/2024/72363023047.csv](https://noaa-global-hourly-pds.s3.amazonaws.com/2024/72363023047.csv)),
while the ISD format specification and ISD-Lite archive live in `noaa-isd-pds`
([https://noaa-isd-pds.s3.amazonaws.com/isd-format-document.pdf](https://noaa-isd-pds.s3.amazonaws.com/isd-format-document.pdf)
and
[https://noaa-isd-pds.s3.amazonaws.com/isd-lite/isd-lite-format.txt](https://noaa-isd-pds.s3.amazonaws.com/isd-lite/isd-lite-format.txt)).
Because WRFComp's training archive (2014–2023) and validation split (2024) are strictly prior to
the sealed-test boundary in `configs/splits.yaml` (`2025-01-01T00:00:00Z`)—and no data at or after
that boundary is ever downloaded, read, or scored—the 2025 ISD wind-down does not affect Sprint 1
verification. If the sealed 2025+ test split is eventually evaluated after human sign-off, a GHCNh
adapter will be required for post-August-2025 dates.

---

## Report-Type Selection (`REPORT_TYPE`)

US ASOS stations in ISD routinely emit aviation METAR/SPECI records (`FM-15` at `:51`–`:56`,
`FM-16` specials) alongside occasional combined synoptic/METAR records (`SY-MT`) that also sit at
routine `:52`–`:54` measurement times and serve as the sole routine report for those hours. In
addition, ASOS stations emit 3-hourly `FM-12` synoptic records at `:00` that duplicate the
preceding `:53` routine measurement with a nominal `:00` timestamp and independent QC flags.
Taking `FM-12` copies alongside routine reports distorts `X_offset_min` (reporting offset `0`
instead of `-7` at synoptic/init hours), can bypass a suspect flag on the underlying routine
observation, and can beat a genuine `:00` `FM-16` special report at the same timestamp.

WRFComp therefore applies a deterministic per-station-year report-type rule:
- **Preferred measurement-time reports:** If a station-year CSV contains any `FM-15`, `FM-16`, or
  `SY-MT` rows, **only** `FM-15`, `FM-16`, and `SY-MT` rows are retained (`FM-12`, `SOD`, `SOM`,
  and other types are dropped).
- **Synoptic fallback:** Only if a station-year has **zero** `FM-15`, `FM-16`, and `SY-MT` rows
  does the loader fall back to `FM-12` rows. Summary rows (`SOD`, `SOM`) are always excluded.

---

## Global-Hourly CSV Fields, Scaling, Range Checks, and Missing Sentinels

All parsing rules and documented ranges follow NOAA NCEI's *Federal Climate Complex Data
Documentation for Integrated Surface Data (ISD)* (`isd-format-document.pdf`,
[https://noaa-isd-pds.s3.amazonaws.com/isd-format-document.pdf](https://noaa-isd-pds.s3.amazonaws.com/isd-format-document.pdf),
pp. 8–12).

| CSV Column | Subfields | Raw Units / Range | Missing Sentinel | Contract Variable & Unit |
| :--- | :--- | :--- | :--- | :--- |
| `TMP` | `<val>,<qc>` | tenths of °C (divide by 10) | `+9999` (or `±9999`) | `t2m` in `K` (`val / 10 + 273.15`) |
| `DEW` | `<val>,<qc>` | tenths of °C (divide by 10) | `+9999` (or `±9999`) | `td2m` in `K` (`val / 10 + 273.15`) |
| `SLP` | `<val>,<qc>` | tenths of hPa (multiply by 10 -> Pa), `> 0` | `99999` | `msl` in `Pa` (`val * 10.0`) |
| `WND` | `<dir>,<dir_qc>,<type>,<spd>,<spd_qc>` | dir: `1..360` deg; spd: `0..900` tenths of m/s (`0..90 m/s`) | dir `999`; spd `9999` | `u10`, `v10` in `m s-1` |

Physical consistency check (per `docs/VERIFICATION_PROTOCOL.md`): wherever both `t2m` and `td2m`
are finite and `td2m > t2m`, `td2m` is masked as `NaN` (and `td2m_offset_min` set to `-32768`).
Stations that do not report sea-level pressure (e.g., AWOS sites reporting altimeter setting only)
remain in the dataset with `msl` as `NaN`.

---

## Passing NOAA Quality Codes

Per `isd-format-document.pdf` (pp. 8–12), quality code domains differ slightly between
temperature/dewpoint and wind/pressure. WRFComp retains only unedited observations whose quality
code is explicitly listed as passing for that field in the format document:

- **Passing set for `TMP` and `DEW` (pp. 10–11):** `{"0", "1", "4", "5", "9", "A", "C"}`
- **Passing set for `WND` (direction and speed, pp. 8–9) and `SLP` (p. 12):** `{"0", "1", "4", "5", "9"}`

Definitions from `isd-format-document.pdf`:
- `0`: Passed gross limits check.
- `1`: Passed all quality control checks.
- `4`: Passed gross limits check, data originate from an NCEI data source.
- `5`: Passed all quality control checks, data originate from an NCEI data source.
- `9`: Passed gross limits check if element is present (accepted **only** when the raw value is not
  a missing sentinel).
- `A`: Data value flagged as suspect, but accepted as a good value (listed on `TMP` and `DEW`).
- `C`: Temperature and dew point received from Automated Weather Observing System (AWOS) are
  reported in whole degrees Celsius; automated QC flags these values, but they are accepted as
  valid (listed on `TMP` and `DEW`; routine `:55` reports on CONUS AWOS stations carry `C`).

Rejected codes:
- `2`, `6`: Suspect.
- `3`, `7`: Erroneous.
- `I` (inserted by validator), `R` (computed by NCEI software): excluded under the "never imputed" rule.
- `M`, `P`, `U` (manual/validator/edited replacements): excluded for strictness.
- `A` on `WND` or `SLP`: not listed in the field's quality-code domain on pp. 8, 9, or 12, so rejected.

---

## Wind Vector Conversion, Calm Winds, and Variable Winds

In ISD, wind direction is reported in meteorological convention: degrees clockwise from true North
indicating the direction the wind comes **FROM** (e.g., 360° = north wind blowing southward,
90° = east wind blowing westward). Eastward (`u10`) and northward (`v10`) wind components are:

```text
rad = deg2rad(direction_deg mod 360)
u10 = -speed_ms * sin(rad)
v10 = -speed_ms * cos(rad)
```

For example, a north wind (360°) at `10 m/s` yields `u10 = 0.0 m/s` and `v10 = -10.0 m/s`.

Special ISD wind type codes (`isd-format-document.pdf`, p. 8):
- **Calm winds (`type == "C"` or `type == "9"` with speed `0000`):** ISD codes calm winds with
  direction `999` and direction QC `9` (e.g., `999,9,C,0000,5`). The ISD format document also notes:
  *"If a value of 9 appears with a wind speed of 0000, this indicates calm winds."* Whenever speed
  is `0000`, speed QC is in `{"0", "1", "4", "5", "9"}`, and `type in {"C", "9"}`, WRFComp assigns
  `u10 = 0.0, v10 = 0.0`.
- **Variable winds (`type == "V"`):** Per p. 8, *"If type code = V, then 999 indicates variable
  wind direction."* Many `V` reports in US ASOS/AWOS files carry a measured direction in `1..360`
  with a passing direction QC (e.g., `190,5,V,0041,5`, where wind direction varied around 190°).
  WRFComp resolves `u10`/`v10` from the reported direction whenever direction is in `1..360` with
  passing direction QC, and sets `u10 = NaN, v10 = NaN` when direction is `999` (e.g.,
  `999,9,V,0026,5`), out of range, or fails QC.

---

## Hourly Matching, Year Boundaries, Observation Time Offsets, and Output Format

- Target grid: every full UTC hour (`datetime64[ns]`, timezone-naive) in `[start, end]`, validated
  via `wrfcomp.harness.schemas.validate_truth_dataset`.
- Tolerance: `±15 minutes` (`evaluation.truth_tolerance_minutes` in `configs/splits.yaml`). Most US
  ASOS/AWOS routine METAR reports arrive at `:51`–`:56` and therefore match the **next** full hour
  (negative offset `-9` to `-4` minutes).
- Year boundaries: both the raw-CSV path and the `--discard-raw` parsed-`.npz` cache path feed the
  matcher the exact same observation window `[start - tol, min(end + tol, sealed_cutoff - 1s)]`,
  automatically including the last 15 minutes of the previous year and the first 15 minutes of the
  next year whenever the window crosses a year boundary (and never reading or downloading anything
  from the sealed test split). Both build paths produce identical output.
- Canonical output windows:
  - Validation truth (`truth/isd_conus_2024.nc`): `2023-12-31T18:00:00` through
    `2024-12-31T23:00:00` UTC (gives persistence baselines a 6-hour causal lookback into 2023 while
    reusing the 2023 station files already needed for training).
  - Training truth (`truth/isd_conus_train.nc`): `2014-01-01T01:00:00` through
    `2023-12-17T18:00:00` UTC (`train.end_utc` in `configs/splits.yaml`). Starting at `01:00Z` on
    2014-01-01 keeps the `±15 min` matching window inside 2014 (`00:45Z` onward) so the 2013 archive
    does not have to be downloaded solely for the 15-minute lookback of `2014-01-01T00:00Z`.
- Matching rule: for each variable independently, pick the QC-passing finite observation with
  minimum `|obs_time - valid_time|` within tolerance. Exact ties pick the **earlier** report.
  Unmatched hours are `NaN` and never interpolated.
- Companion offset variables (Addendum 1): every primary variable `X` carries `X_offset_min`
  (`int16`, dims `(valid_time, station_id)`), storing `measurement_time - valid_time` in whole
  minutes (`-15..+15`), with `-32768` wherever `X` is `NaN`. Files are written as NetCDF4 via
  `h5netcdf` with no `_FillValue` attribute or encoding on `X_offset_min` so `xarray.open_dataset`
  always preserves plain `int16`. Downstream causal consumers must use an observation only when
  `valid_time + X_offset_min <= init_time`.
