# Citizen Weather Station Data Sources & 2024 Sample (`M1`)

This document summarizes our live investigation of free, open sources for 2024 CWOP / APRSWXNET
citizen-station surface observations for the `wrfcomp.assimilation` quality-control trial, and records
aggregate statistics for the anonymized 2024-07-08 to 2024-07-21 sample around Amarillo, TX and
Seattle, WA.

> **Privacy rule:** Citizen stations are located at private residences. No citizen station identifier,
> call sign, station name, or exact coordinate is printed, logged, or committed. Data on disk holds a
> keyed 11-character station token (`CW<9 hex>`), latitude/longitude rounded to `0.1` degree and
> elevation rounded to `10 m`. This document reports counts and aggregates only.

---

## 1. Source Survey (Verified by Live Fetching, 2026-11-23)

Verification status: the selected source (first row) was fetched again when the lane was finished: the
sample in section 3 came from it with no account, token or terms step, and the "No restrictions" wording
was re-read on `https://madis.ncep.noaa.gov/madis_cwop.shtml` (page dated 02 December 2021). The other
rows are as recorded in the first pass of this lane and were not fetched a second time.

| Source | URL(s) Probed | Account / Token Required? | 2024 CWOP / APRSWXNET Available? | Licence / Redistribution Limits | Format & Time Resolution | Station Metadata Available |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NOAA MADIS Public Surface CGI (`madisPublic1`)** *(Selected)* | `https://madis-data.ncep.noaa.gov/madisPublic1/cgi-bin/madisXmlPublicDir` (form: `https://madis-data.ncep.noaa.gov/public/sfcdumpguest.html`) | **No** (open anonymous HTTPS GET, no account, email, token, or click-through terms) | **Yes** (`pvdrsel=1&pvd=APRSWXNET`, full 2024 archive verified) | **No restrictions** for CWOP / `APRSWXNET` per `https://madis.ncep.noaa.gov/madis_cwop.shtml`: *"No restrictions. All observations are publicly accessible."* (Non-CWOP mesonet providers in MADIS carry redistribution restrictions per `madis_restrictions.shtml`, so we filter strictly to `APRSWXNET`.) | HTML-wrapped CSV (`xml=5`) or XML (`xml=1`); SI units (`T`/`TD` K, `FF`/`FFGUST` m/s, `DD` deg, `P`/`SLP` Pa, `PCP1H` m); **exact sub-hourly observation times (`OBDATE` + `OBTIME` to the minute, `recwin=0`, typically every 5–15 min)** | Inline `LAT`, `LON`, `ELEV` (m) on 100% of rows via `nvars=LAT,LON,ELEV`; current provider table also open at `madisPublic1/data/stations/APRSWXNETStation.txt` (13,397 rows, 1.2 MiB, pipe-delimited, omits stations retired since 2024) |
| **NOAA MADIS Public Bulk NetCDF Archive (`madisPublic1`)** | `https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/2024/07/08/LDAD/mesonet/netCDF/20240708_0000.gz` | **No** (open anonymous HTTPS directory tree; `madisResearch` returns HTTP 403 Forbidden) | **Yes** (`APRSWXNET` is the largest `dataProvider` inside `LDAD/mesonet/netCDF/`: ~70,740 reports/hr globally; `point/` has METAR/SAO/maritime/upper-air, not CWOP) | Same as above: `APRSWXNET` is unrestricted public distribution; other providers bundled in the same global NetCDF file (e.g. `MesoWest`, state DOTs) carry non-redistribution terms. | Gzipped NetCDF-3 (`~25.5 MiB/hr` compressed, `~361 MiB/hr` uncompressed -> **~8.4 GiB** for 14 days, exceeding the 500 MB sample cap); exact `observationTime` (`datetime64[ns]`) | Full inline `latitude`, `longitude`, `elevation`, `dataProvider`, `stationId`, MADIS QC flags |
| **Synoptic Data / MesoWest API** | `https://api.synopticdata.com/v2/stations/metadata?state=TX&network=65` | **Yes** (HTTP 401 `Missing token`; requires registered account, API token, and Terms of Service acceptance; free tier is limited to academic/non-commercial Open Access) | Yes (with account/token) | Synoptic Terms of Service govern API use; public-data redistribution requires attribution and compliance with upstream provider limits. | JSON/CSV REST API; sub-hourly | Full station metadata via `/v2/stations/metadata` |
| **FindU / APRS-IS** | `http://www.findu.com/cgi-bin/raw.cgi?call=...` (HTTP 404), `http://www.findu.com/cgi-bin/wxpage.cgi?...` (HTTP 200) | No for recent single-station queries | **No** (public web/CGI historical window is capped to the last ~10 days; no bulk 2024 archive download endpoint) | Amateur radio / volunteer APRS public feed; no bulk redistribution API | HTML/text per single callsign; sub-hourly | Callsign/position embedded in APRS packets |
| **NOAA NODD S3 Buckets** | `https://noaa-madis-pds.s3.amazonaws.com/?list-type=2&max-keys=5` | No (if a bucket existed) | **No** (`NoSuchBucket` HTTP 404; MADIS / CWOP is hosted on `madis-data.ncep.noaa.gov`, not on AWS NODD S3) | N/A | N/A | N/A |
| **Iowa Environmental Mesonet (IEM)** | `https://mesonet.agron.iastate.edu/geojson/networks.geojson` (HTTP 200, 599 networks) | No (open public archive) | **No** for CWOP / APRSWXNET (IEM indexes ASOS/AWOS, RWIS, DCP, and CoCoRaHS daily precip, not CWOP/APRSWXNET hourly/sub-hourly surface networks) | Open public research archive | CSV/GeoJSON | Full station metadata per network |

### Why NOAA MADIS Public Surface CGI (`madisXmlPublicDir`) Was Chosen

1. **Zero authentication or terms barriers:** Directly accessible over anonymous HTTPS with no account, email, API token, or terms acceptance.
2. **Unrestricted CWOP subset only:** Filtering server-side with `pvdrsel=1&pvd=APRSWXNET` excludes restricted third-party mesonets and pulls only NOAA's unrestricted public CWOP citizen observations.
3. **Server-side spatial subsetting keeps transfers small:** Downloading raw global hourly MADIS NetCDF files for 14 days (336 hours) would transfer `~8.4 GiB`, exceeding the 500 MB cap. Server-side bounding-box subsetting via `madisXmlPublicDir` transfers only `~149 MiB` for all 336 hours across both target regions, with exact historical 2024 `LAT`, `LON`, `ELEV` inline on every row.
4. **Raw observations preserved for our own QC:** Passing `qcsel=0` returns all observations regardless of MADIS QC flags, and `recwin=0` preserves every sub-hourly report at its exact UTC minute (`OBDATE` + `OBTIME`, never rounded to the hour).

---

## 2. Loader & Privacy Implementation

**Loader:** `src/wrfcomp/assimilation/citizen_obs.py`

- `parse_madis_csv` reads one response (with or without the `<pre>` wrapper), keeps `PVDR == "APRSWXNET"`
  rows, applies the 150 km radius on the exact position in memory, and returns an `ObservationBatch`
  (`sensor_type_id = 2`) that the QC code takes as is.
- **Units.** MADIS serves SI units, so values pass through: `T -> t2m` [K], `TD -> td2m` [K],
  `FF -> wind_speed` [m/s], `FFGUST -> wind_gust` [m/s], `P -> station_pressure` [Pa], `SLP -> msl` [Pa].
  Two conversions only: `(DD, FF) -> u10, v10` with `DD` the direction the wind comes FROM
  (`u = -FF sin DD`, `v = -FF cos DD`), and `PCP1H` from m to mm (`precip`).
- **Times.** `OBDATE` + `OBTIME` are the measurement time in UTC, to the minute. They are kept as they
  are and never moved to the hour.
- **MADIS quality flags are not used.** The request asks for all reports (`qcsel=0`) and the descriptor
  columns are not read: the purpose of the sample is to run our own quality control on raw reports.
- **Duplicates.** One value per (station, time, variable); the last row wins.
- **Sealed split.** Any time at or after the sealed-test start in `configs/splits.yaml` is refused, in the
  parser, in the merge, on save and on load.

**Privacy, by construction**

- **Station tokens are keyed.** The raw identifier is replaced inside the parser by `CW` plus 9 hex digits
  of an HMAC-SHA256. The key is 32 random bytes created on first use in
  `<data dir>/citizen/.station_key` (mode 0600). It is not in the repository. A plain hash with a salt
  from the code would not protect anyone, because the list of CWOP identifiers is small and public and
  could simply be hashed and compared; with the key held only on this machine that does not work.
  Consequences: the same station has the same token on this machine only, the key file must never be
  committed, printed or shipped with data, and deleting it breaks the link to older files.
- **Positions are rounded in the parser,** not only on save: latitude/longitude to 0.1 degree, elevation
  to 10 m. No batch, file or summary holds an exact position, and the parser has no option to return one.
  `save_citizen_batch` rounds again, so a batch built by other code cannot put an exact position on disk
  through it. Elevation is rounded because an exact elevation narrows down a point inside a 0.1 degree
  cell in hilly country; 10 m changes the lapse-rate adjustment in the buddy check by about 0.07 K.
- **Nothing identifying in logs.** A row that cannot be read is skipped, never quoted. The fetch script
  prints region names, window times, counts and exception class names; it never prints an exception
  message, because a message could quote a row.
- **Raw responses never touch disk.** They are parsed in memory; only the tokenised, rounded NetCDF and an
  aggregate summary JSON are written.

**Fetcher:** `scripts/fetch_citizen_sample.py`

- 2-hour windows per region, at most `4` concurrent requests, dates before the sealed-test start only.
- The `500 MiB` hard cap applies twice: to the bytes transferred (the run aborts when the running total
  passes it) and to the bytes stored (after each region is saved the `*.nc` files in the folder are
  added up; a region that would take the total over the cap is deleted again and the run fails).
- Storage: one NetCDF4 file per region, float32 values, station token and variable name as fixed-width
  byte strings, zlib compression. A first version of this lane wrote the same rows uncompressed with
  wide text columns, about 123 bytes per row (573 MB for the Seattle region); the present layout takes
  about 5 bytes per row. `--sample-windows N` fetches N windows and prints a size projection without
  writing data.
- A window that still fails after three tries is counted in `errors` in the summary; the sample is then
  incomplete for that window and section 3 says so.

**Limits of the sample as stored**

- Buddy distances are computed from positions on a 0.1 degree grid (about 11 km in latitude), so stations
  in the same cell look co-located and distances under about 15 km are coarse. The QC trial should treat
  the 25 km buddy radius accordingly.
- Station tokens cannot be joined to any outside station list, by design.

---

## 3. 2024-07-08 to 2024-07-21 Sample Summary (Counts & Aggregates Only)

*(Pending completion of full 14-day sample fetch.)*
