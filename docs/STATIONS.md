# Vetted CONUS Reference Stations (`configs/stations_conus.csv`)

This document records the 2024 NOAA ISD-Lite availability vetting that produced the 500-station
CONUS reference list in `configs/stations_conus.csv` via `scripts/vet_stations.py`.

## Data Source & Access Notes

- **Station catalog:** `https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv` (fetched 2026-11-23).
- **2024 hourly counts (for vetting only, never for ground-truth scoring):** NOAA ISD-Lite 2024
  archives. Note that `https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/2024/<USAF>-<WBAN>-2024.gz`
  redirects with HTTP 200 HTML to `https://noaa-isd-pds.s3.amazonaws.com/index.html#isd-lite/`.
  The official NOAA ISD-Lite 2024 `.gz` files are hosted directly in the same NOAA Open Data
  Dissemination bucket at
  `https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/2024/<USAF>-<WBAN>-2024.gz`.
- **Download footprint:** At 4 concurrent requests, a 10-station dry-run sample measured
  651,148 bytes (635.9 KiB total, 63.6 KiB/station, projecting ~142.2 MiB for 2,290 candidates).
  The full vetting run downloaded/checked all 2,290 candidates in 279.2 s: 2,254 files totaling
  162,881,902 bytes (155.3 MiB), 36 HTTP 404 responses (no 2024 ISD-Lite file), and 0 transfer errors.
- **No 2025+ data accessed:** Only 2024 ISD-Lite files were downloaded and parsed.

## Vetting Rules

1. **CONUS geography & active span** (same base filter as `scripts/fetch_conus_stations.py`, with
   the 2014–2023 history requirement):
   - `CTRY == "US"`, `STATE` not in `{AK, HI, PR, VI, GU, AS, MP}`, `LAT in [24.5, 49.5]`,
     `LON in [-125.0, -66.5]`, non-null `LAT`/`LON`/`ELEV(M)`/`BEGIN`/`END`, `USAF != "999999"`.
   - `BEGIN <= 20140101` (full 2014–2023 training history per `isd-history.csv`).
   - `END >= 20241231` (active through the end of 2024).
2. **2024 hourly completeness** (2024 is a leap year: 366 days = 8,784 hours; `-9999` is missing;
   calm wind direction `0` is valid per NOAA's ISD-Lite format specification):
   - Air temperature (`temp`): `>= 90%` (`>= 7,906 / 8,784` hours)
   - Dew point temperature (`dewpoint`): `>= 90%` (`>= 7,906 / 8,784` hours)
   - Wind speed (`wind_speed`): `>= 90%` (`>= 7,906 / 8,784` hours)
   - Wind direction (`wind_dir`): `>= 90%` (`>= 7,906 / 8,784` hours)
   - Sea level pressure (`slp`): `>= 80%` (`>= 7,028 / 8,784` hours)
   - Maximum gap: no contiguous missing run longer than 10 days (`<= 240` hours) in any of the
     five required variables (including leading/trailing gaps at the start/end of 2024).
3. **Spatial selection:** Deterministic greedy farthest-point sampling reusing
   `_select_spatially_diverse` from `scripts/fetch_conus_stations.py` (prefers real WBAN
   `!= "99999"` when at least 500 such stations pass; 833 passing stations have a real WBAN, so all
   500 selected sites are primary WBAN stations).

## Candidate & Rule Pass Counts

| Stage / Rule | Standalone Pass | Sequential Remaining |
| :--- | ---: | ---: |
| CONUS active through 2024 (`BEGIN <= 20200101`, `END >= 20241231`) | — | 2,376 |
| **Candidates with 2014–2023 history (`BEGIN <= 20140101`)** | **2,290** | **2,290** |
| 2024 ISD-Lite file present (non-404) | 2,254 | 2,254 |
| Air temperature `>= 90%` of 8,784 h | 1,670 | 1,670 |
| Dew point `>= 90%` of 8,784 h | 1,607 | 1,606 |
| Wind speed `>= 90%` of 8,784 h | 1,806 | 1,572 |
| Wind direction `>= 90%` of 8,784 h | 1,605 | 1,390 |
| Sea level pressure `>= 80%` of 8,784 h | 1,303 | 911 |
| No gap `> 10 days` (`<= 240 h`) across all 5 required variables | 1,028 | **874** |
| **Final spatially diverse selection (`configs/stations_conus.csv`)** | — | **500** |

*(Diagnostic note: if the 10-day gap rule were applied only to raw record presence rather than each
of the 5 required variables, 1,930 candidates pass standalone and 902 pass all rules. We enforce the
stricter per-variable 10-day gap rule, yielding 874 passing stations.)*

## Provisional 500 Survival

Comparing the previous provisional `configs/stations_conus.csv` (picked from `isd-history.csv` by
spacing alone with `BEGIN <= 20200101`) against the 2024 vetting results:

- Provisional stations: **500**
- Met `BEGIN <= 20140101` history filter: **491 / 500 (98.2%)**
- Passed all 2024 ISD-Lite completeness and gap rules: **232 / 500 (46.4%)**
- Survived into the final re-picked 500 (`configs/stations_conus.csv`): **188 / 500 (37.6%)**

Over half of the provisional 500 failed vetting—predominantly small AWOS sites that do not report
sea-level pressure (`SLP = -9999` for all hours in 2024).

## Per-State Counts

The 500 vetted stations cover 47 contiguous US states (Rhode Island had 4 passing sites, none picked
by farthest-point sampling because neighboring Massachusetts and Connecticut sites were closer to
the greedy frontier; 104 candidates and 17 passing stations had blank `STATE` in `isd-history.csv`,
none of which were selected).

| State | Candidates (`BEGIN <= 20140101`) | Passing All Rules | Selected (500) |
| :--- | ---: | ---: | ---: |
| AL | 39 | 16 | 10 |
| AR | 34 | 15 | 11 |
| AZ | 36 | 16 | 11 |
| CA | 151 | 74 | 33 |
| CO | 60 | 27 | 15 |
| CT | 10 | 8 | 2 |
| DE | 6 | 3 | 1 |
| FL | 123 | 40 | 18 |
| GA | 64 | 14 | 8 |
| IA | 59 | 15 | 10 |
| ID | 24 | 14 | 10 |
| IL | 58 | 18 | 11 |
| IN | 32 | 13 | 10 |
| KS | 35 | 22 | 14 |
| KY | 23 | 10 | 6 |
| LA | 51 | 13 | 9 |
| MA | 26 | 16 | 5 |
| MD | 30 | 8 | 1 |
| ME | 22 | 8 | 5 |
| MI | 105 | 34 | 20 |
| MN | 94 | 16 | 13 |
| MO | 32 | 19 | 11 |
| MS | 33 | 15 | 10 |
| MT | 36 | 20 | 16 |
| NC | 72 | 22 | 13 |
| ND | 41 | 11 | 9 |
| NE | 46 | 18 | 13 |
| NH | 14 | 5 | 3 |
| NJ | 19 | 6 | 3 |
| NM | 32 | 16 | 15 |
| NV | 20 | 13 | 9 |
| NY | 42 | 27 | 12 |
| OH | 46 | 25 | 9 |
| OK | 52 | 15 | 10 |
| OR | 36 | 17 | 12 |
| PA | 36 | 20 | 9 |
| RI | 9 | 4 | 0 |
| SC | 51 | 16 | 8 |
| SD | 22 | 15 | 12 |
| TN | 19 | 9 | 5 |
| TX | 195 | 64 | 38 |
| UT | 19 | 11 | 9 |
| VA | 58 | 14 | 5 |
| VT | 11 | 4 | 1 |
| WA | 53 | 30 | 14 |
| WI | 65 | 19 | 11 |
| WV | 15 | 8 | 7 |
| WY | 30 | 14 | 13 |
| *(blank)* | 104 | 17 | 0 |
| **Total** | **2,290** | **874** | **500** |
