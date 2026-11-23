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
2025-01-01—and no data from 2025-01-01 onward is ever downloaded, read, or scored—the 2025 ISD
wind-down does not affect Sprint 1 verification. If the sealed 2025+ test split is eventually
evaluated after human sign-off, a GHCNh adapter will be required for post-August-2025 dates.

---

## Global-Hourly CSV Fields, Scaling, and Missing Sentinels

All parsing rules follow NOAA NCEI's *Federal Climate Complex Data Documentation for Integrated
Surface Data (ISD)* (`isd-format-document.pdf`,
[https://noaa-isd-pds.s3.amazonaws.com/isd-format-document.pdf](https://noaa-isd-pds.s3.amazonaws.com/isd-format-document.pdf),
pp. 7–12).

| CSV Column | Subfields | Raw Units / Scaling | Missing Sentinel | Contract Variable & Unit |
| :--- | :--- | :--- | :--- | :--- |
| `TMP` | `<val>,<qc>` | tenths of °C (divide by 10) | `+9999` (or `±9999`) | `t2m` in `K` (`val / 10 + 273.15`) |
| `DEW` | `<val>,<qc>` | tenths of °C (divide by 10) | `+9999` (or `±9999`) | `td2m` in `K` (`val / 10 + 273.15`) |
| `SLP` | `<val>,<qc>` | tenths of hPa (multiply by 10 -> Pa) | `99999` | `msl` in `Pa` (`val * 10.0`) |
| `WND` | `<dir>,<dir_qc>,<type>,<spd>,<spd_qc>` | dir: degrees (0–360); spd: tenths of m/s (divide by 10) | dir `999`; spd `9999` | `u10`, `v10` in `m s-1` |

Physical consistency check (per `docs/VERIFICATION_PROTOCOL.md`): wherever both `t2m` and `td2m`
are finite and `td2m > t2m`, `td2m` is masked as `NaN` (and `td2m_offset_min` set to `-32768`).

---

## Passing NOAA Quality Codes

Per `isd-format-document.pdf` (pp. 8–12), each mandatory parameter carries a single-character
quality code. WRFComp retains only unedited observations that passed NOAA quality control:

- **Passing set for `TMP` and `DEW`:** `{"0", "1", "4", "5", "9", "A", "C"}`
- **Passing set for `SLP` and `WND`:** `{"0", "1", "4", "5", "9", "A"}`

Definitions from `isd-format-document.pdf`:
- `0`: Passed gross limits check.
- `1`: Passed all quality control checks.
- `4`: Passed gross limits check, data originate from an NCEI data source.
- `5`: Passed all quality control checks, data originate from an NCEI data source.
- `9`: Passed gross limits check if element is present (accepted **only** when the raw value is not
  a missing sentinel).
- `A`: Data value flagged as suspect, but accepted as a good value (NCEI Interactive QC).
- `C`: Temperature and dew point received from Automated Weather Observing System (AWOS) are
  reported in whole degrees Celsius; automated QC flags these values, but they are accepted as
  valid. (In US CONUS AWOS stations, routine `:55` hourly reports carry `C`.)

Rejected codes:
- `2`, `6`: Suspect.
- `3`, `7`: Erroneous.
- `I` (inserted by validator), `R` (computed by NCEI software): excluded under the "never imputed" rule.
- `M`, `P`, `U` (manual/validator/edited replacements): excluded for strictness.

---

## Wind Vector Conversion, Calm Winds, and Variable Winds

In ISD, wind direction is reported in meteorological convention: degrees clockwise from true North
indicating the direction the wind comes **FROM** (e.g., 0° / 360° = north wind blowing southward,
90° = east wind blowing westward). Eastward (`u10`) and northward (`v10`) wind components are:

```text
rad = deg2rad(direction_deg mod 360)
u10 = -speed_ms * sin(rad)
v10 = -speed_ms * cos(rad)
```

For example, a north wind (0° or 360°) at `10 m/s` yields `u10 = 0.0 m/s` and `v10 = -10.0 m/s`.

Special ISD wind type codes (`isd-format-document.pdf`, p. 8):
- **Calm winds (`type == "C"` or `type == "9"` with speed `0000`):** ISD codes calm winds with
  direction `999` and direction QC `9` (e.g. `999,9,C,0000,5`). The ISD format document also notes:
  *"If a value of 9 appears with a wind speed of 0000, this indicates calm winds."* Whenever speed
  is `0000`, speed QC is in the passing set, and `type in {"C", "9"}`, WRFComp assigns
  `u10 = 0.0, v10 = 0.0`.
- **Variable winds (`type == "V"`):** Wind direction is variable and coded as `999` (e.g.
  `999,9,V,0021,5`). Because horizontal vector components cannot be resolved without direction,
  both `u10` and `v10` are set to `NaN`.

---

## Hourly Matching and Observation Time Offsets

- Target grid: every full UTC hour (`datetime64[ns]`, timezone-naive) in `[start, end]`.
- Tolerance: `±15 minutes` (`evaluation.truth_tolerance_minutes` in `configs/splits.yaml`). Most US
  ASOS/AWOS routine METAR reports arrive at `:51`–`:56` and therefore match the **next** full hour
  (negative offset `-9` to `-4` minutes).
- Matching rule: for each variable independently, pick the QC-passing finite observation with
  minimum `|obs_time - valid_time|` within tolerance. Exact ties pick the **earlier** report.
  Summary-of-day (`SOD`) and summary-of-month (`SOM`) records at `:59` contain only missing
  sentinels and are therefore naturally skipped. Unmatched hours are `NaN` and never interpolated.
- Companion offset variables (Addendum 1): every primary variable `X` carries `X_offset_min`
  (`int16`, dims `(valid_time, station_id)`), storing `measurement_time - valid_time` in whole
  minutes (`-15..+15`), with sentinel `-32768` wherever `X` is `NaN`. Downstream causal consumers
  (such as persistence baselines or assimilation) must use an observation only when
  `valid_time + X_offset_min <= init_time`.
