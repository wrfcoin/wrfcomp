# Scoring Data Bundles

To score a candidate forecast locally with `scripts/check_promotion.py` against the locked harness
(`harness-v0.1.0`), you need the exact datasets pinned in `experiments/REFERENCES.json`:

1. **Station ground truth** (`truth/isd_conus_<year>.nc`) — hourly NOAA ISD station observations for
   the 500 CONUS stations in `configs/stations_conus.csv`.
2. **Tail thresholds** (`thresholds/tail_p95.nc`) — per-station, per-calendar-month 95th-percentile
   thresholds from the training split.
3. **Pinned reference forecast stores** (`gate1/<period>_h00-12/climatology.zarr` and `gefs.zarr`) —
   every monthly and quarterly reference store named in `experiments/REFERENCES.json` for that year.

These datasets are distributed as one per-year `.tar.gz` archive on Zenodo (one free, citable
record per year). **Nothing from the sealed test split (2025 onward) and no citizen weather-station
data is ever included.**

---

## How to Get the Scoring Data

Set `WRFCOMP_DATA_DIR` if you want the data stored somewhere other than the default
`~/wrfcomp-data/`, then run `scripts/fetch_scoring_data.py` with the Zenodo record URL (or direct
archive URL) and the trusted archive SHA-256 (or `.sha256` / `.MANIFEST.json` sidecar) published
for that year:

```bash
uv run python -P scripts/fetch_scoring_data.py \
  --url <ZENODO_RECORD_URL> \
  --expected-sha256 <ARCHIVE_SHA256> \
  --year 2024
```

You can also pass `--expected-sha256-file <PATH>` or `--expected-manifest <PATH>` for the trusted
physical pin, or `--archive <PATH>` to verify and unpack an archive you already downloaded:

```bash
uv run python -P scripts/fetch_scoring_data.py \
  --archive wrfcomp-scoring-2024.tar.gz \
  --expected-sha256 <ARCHIVE_SHA256> \
  --year 2024
```

### What `fetch_scoring_data.py` Verifies

Before anything is moved into your data directory, the script unpacks into a private staging
directory and checks:

1. **Physical archive pinning:** requires an independently trusted `--expected-sha256`,
   `--expected-sha256-file`, or `--expected-manifest` by default, and checks `MANIFEST.json`'s
   `references_sha256` against the repository's `experiments/REFERENCES.json`.
2. **Bounded streaming download & extraction:** enforces download size, member count, per-member
   size, manifest size, total expanded bytes, staging free space, and decoded Zarr chunk/dataset
   size limits before hashing.
3. **Archive & destination safety:** rejects POSIX and Windows path traversal (`..`, drive letters,
   colons, backslashes, reserved device names), symlinks and hardlinks (at staging, source, and
   final destination ancestors), special files, forbidden directories (such as `citizen/`), and any
   unmanifested or unrelated files inside Zarr stores.
4. **`MANIFEST.json` & strict Zarr layout:** verifies every member file's exact byte size and byte
   SHA-256 and enforces strict Zarr v2 metadata/chunk-grid layout before opening with `xarray`.
5. **Repository pins (`experiments/REFERENCES.json`):** opens every NetCDF and Zarr dataset with
   `xarray`, verifies that no timestamps reach the sealed split in `configs/splits.yaml`, and checks
   that the harness `content_sha256` matches **both** `MANIFEST.json` and
   `experiments/REFERENCES.json`.
6. **Atomic install:** moves verified datasets into `$WRFCOMP_DATA_DIR` only after all checks pass
   and destination ancestors are confirmed symlink-free. Existing files that already match the
   pinned hash are left untouched; existing files with different content are refused unless
   `--overwrite` is passed.

Once unpacked, you can run the locked promotion check immediately:

```bash
uv run --locked python -P scripts/check_promotion.py evaluate \
  --forecast <2024-01.zarr> <2024-02.zarr> <2024-03.zarr> \
  --data-dir "${WRFCOMP_DATA_DIR:-$HOME/wrfcomp-data}" \
  --period 2024-Q1 \
  --out <out-dir>
```

---

## Building a Bundle (Maintainer / Release Workflow)

The repository owner builds each yearly bundle from verified local inputs and uploads it to Zenodo:

```bash
uv run python -P scripts/build_scoring_bundle.py --year 2024
```

`scripts/build_scoring_bundle.py`:

- Checks that at least 40 GiB of free disk space will remain on the output filesystem
  (`--min-free-gb 40`).
- Reads `experiments/REFERENCES.json` and collects **only** the pinned truth, thresholds, and
  reference forecast stores for `--year` (refusing sealed years `>= 2025` and unpinned years such
  as `2023` until its references are pinned).
- Verifies every dataset's harness `content_sha256` against `experiments/REFERENCES.json` **before**
  packing, refusing on any mismatch.
- Packs a deterministic archive (`wrfcomp-scoring-<year>.tar.gz`, normalized tar metadata and
  `mtime=0`) containing `MANIFEST.json`, a bundle `README.md`, and the pinned datasets under
  `<data-dir>/bundle/`, alongside `.sha256` and `.MANIFEST.json` sidecar files.

### Sources and License Recommendation

- **Upstream sources (public domain):**
  - Ground truth (`truth/`), tail thresholds (`thresholds/`), and climatology references
    (`climatology.zarr`) are derived from the NOAA NCEI Integrated Surface Database (ISD) Global
    Hourly archive (see [docs/TRUTH_SOURCES.md](TRUTH_SOURCES.md)).
  - GEFS references (`gefs.zarr`) are derived from NOAA Global Ensemble Forecast System (GEFS v12)
    forecasts via NOAA Open Data Dissemination.
  - Both NOAA sources are U.S. federal government works in the public domain under 17 U.S.C. § 105.
- **Derived bundle license recommendation — `CC0-1.0`:**
  We recommend publishing the Zenodo record under **CC0-1.0 (Public Domain Dedication)** rather
  than CC-BY-4.0. The upstream NOAA archives are already public domain, and the bundled files are
  deterministic numerical extractions (nearest-hour matching, station-point interpolation, and
  empirical quantiles). `CC0-1.0` prevents attribution-stacking friction in automated verification
  and CI workflows while the Zenodo DOI still provides standard scholarly citation metadata.
  (`CC-BY-4.0` is an acceptable alternative if formal attribution is preferred.)

---

## Availability by Year

| Year | Status | Pinned periods in `experiments/REFERENCES.json` | Zenodo URL |
| --- | --- | --- | --- |
| `2024` | Bundle built; awaiting owner Zenodo upload | `2024-01`..`2024-12`, `2024-Q1`..`2024-Q4` (34 datasets) | *Pending owner upload* |
| `2023` | Planned after 2023 references are pinned | — | — |
