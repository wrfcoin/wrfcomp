"""Build a verifiable, reproducible per-year scoring data bundle for Zenodo archival.

Reads ``experiments/REFERENCES.json``, collects *only* the files and Zarr stores pinned for
``--year`` (truth, p95 tail thresholds, and the pinned monthly/quarterly reference stores),
verifies every dataset's harness ``content_sha256`` against the pins *before* packing, writes
``MANIFEST.json`` and a short ``README.md`` inside the archive, and writes one reproducible
``.tar.gz`` archive per year under ``<data-dir>/bundle/`` (plus ``.sha256`` and
``.MANIFEST.json`` sidecars).

Nothing from the sealed split (2025 onward) and no citizen data, candidate stores, or raw caches
are ever read or included.

Usage:
    uv run python -P scripts/build_scoring_bundle.py --year 2024
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from wrfcomp.harness.align import check_forecast_not_sealed, check_not_sealed
from wrfcomp.harness.config import HarnessConfig, load_config
from wrfcomp.harness.hashing import content_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = ROOT / "experiments" / "REFERENCES.json"
DEFAULT_CONFIG = ROOT / "configs" / "splits.yaml"
DEFAULT_MIN_FREE_GB = 40.0
DEFAULT_COMPRESS_LEVEL = 1
MANIFEST_VERSION = 1
RECOMMENDED_LICENSE = "CC0-1.0"
REQUIRED_BASELINES = ("climatology", "noaa-gefs")
FORBIDDEN_PATH_PARTS = frozenset(
    {
        "citizen",
        ".station_key",
        "candidates",
        "baselines",
        "raw",
        "parsed",
        "bundle",
    }
)
PERIOD_RE = re.compile(r"^(?P<year>[0-9]{4})-(?:[0-9]{2}|Q[1-4])$")


class BundleError(RuntimeError):
    """Raised when bundle inputs, pins, disk space, or hashes fail validation."""


@dataclass(frozen=True)
class PinnedItem:
    relpath: str
    kind: str  # "truth" | "thresholds" | "reference_forecast"
    expected_sha256: str
    pinned_by: tuple[dict[str, str], ...]


def default_data_dir() -> Path:
    env = os.environ.get("WRFCOMP_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "wrfcomp-data"


def _validate_relpath(relpath: str, what: str) -> Path:
    if not isinstance(relpath, str) or not relpath:
        raise BundleError(f"{what}: empty or non-string relative path")
    if relpath.startswith("/") or "\\" in relpath:
        raise BundleError(f"{what}: path must be a relative POSIX path, got {relpath!r}")
    p = Path(relpath)
    if p.is_absolute() or ".." in p.parts or "." in p.parts:
        raise BundleError(f"{what}: unsafe relative path {relpath!r}")
    lowered = {part.lower() for part in p.parts}
    forbidden = lowered & FORBIDDEN_PATH_PARTS
    if forbidden:
        raise BundleError(f"{what}: forbidden path component {sorted(forbidden)} in {relpath!r}")
    return p


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _open_dataset(path: Path, what: str) -> xr.Dataset:
    if not path.exists():
        raise BundleError(f"{what}: missing path {path.name}")
    if path.is_symlink():
        raise BundleError(f"{what}: symlinks are refused ({path.name})")
    try:
        if path.is_dir():
            try:
                return xr.open_zarr(path, chunks=None, consolidated=True)
            except (KeyError, ValueError, FileNotFoundError):
                return xr.open_zarr(path, chunks=None, consolidated=False)
        return xr.open_dataset(path)
    except Exception as exc:
        raise BundleError(f"{what}: failed to open {path.name}: {exc}") from exc


def _check_not_sealed(ds: xr.Dataset, kind: str, relpath: str, cfg: HarnessConfig) -> None:
    sealed_start = cfg.splits.sealed_test.start
    if kind == "truth":
        if "valid_time" not in ds.coords:
            raise BundleError(f"{relpath}: truth dataset missing valid_time coordinate")
        check_not_sealed(ds["valid_time"].values, what=f"truth {relpath}", sealed_start=sealed_start)
    elif kind == "reference_forecast":
        check_forecast_not_sealed(ds, what=f"reference {relpath}", sealed_start=sealed_start)


def collect_pinned_items(
    references_path: Path,
    year: int,
    cfg: HarnessConfig,
) -> tuple[dict[str, Any], list[str], list[PinnedItem]]:
    """Read ``REFERENCES.json`` and return ``(inventory, periods, pinned_items)`` for ``year``."""
    sealed_dt = cfg.splits.sealed_test.start_utc
    sealed_year = sealed_dt.year
    sealed_iso = sealed_dt.replace(tzinfo=None).isoformat()
    if year >= sealed_year:
        raise BundleError(
            f"Refusing year {year}: sealed split starts at {sealed_iso}"
        )
    try:
        inventory = json.loads(references_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"Cannot read references inventory: {exc}") from exc
    if not isinstance(inventory, dict) or not isinstance(inventory.get("periods"), dict):
        raise BundleError("Invalid REFERENCES.json structure")

    periods: list[str] = []
    # Map relpath -> (kind, expected_sha256, list[pin_citation])
    seen: dict[str, tuple[str, str, list[dict[str, str]]]] = {}

    def record(relpath: str, kind: str, digest: str, citation: dict[str, str]) -> None:
        _validate_relpath(relpath, f"{kind} ({citation['period']})")
        if not isinstance(digest, str) or len(digest) != 64:
            raise BundleError(f"Invalid sha256 pin for {relpath}: {digest!r}")
        if relpath in seen:
            prev_kind, prev_digest, citations = seen[relpath]
            if prev_kind != kind:
                raise BundleError(f"Conflicting kinds for {relpath}: {prev_kind} vs {kind}")
            if prev_digest != digest:
                raise BundleError(f"Conflicting sha256 pins for {relpath}: {prev_digest} vs {digest}")
            citations.append(citation)
        else:
            seen[relpath] = (kind, digest, [citation])

    for period in sorted(inventory["periods"]):
        m = PERIOD_RE.fullmatch(period)
        if not m or int(m.group("year")) != year:
            continue
        periods.append(period)
        entry = inventory["periods"][period]
        refs = entry.get("references")
        if not isinstance(refs, dict) or set(refs) != set(REQUIRED_BASELINES):
            raise BundleError(
                f"Period {period} must pin exactly {', '.join(REQUIRED_BASELINES)}"
            )
        truth_rel = entry["truth"]
        thresh_rel = entry["thresholds"]
        for baseline in REQUIRED_BASELINES:
            rec = refs[baseline]
            for time_key in ("period_start", "period_end"):
                stamp = str(rec.get(time_key, ""))
                if stamp >= sealed_iso:
                    raise BundleError(f"Period {period} {baseline} {time_key} reaches sealed split: {stamp}")
            scorecard = str(rec["scorecard"])
            record(
                truth_rel,
                "truth",
                rec["truth_sha256"],
                {
                    "period": period,
                    "baseline": baseline,
                    "field": "truth_sha256",
                    "scorecard": scorecard,
                },
            )
            record(
                thresh_rel,
                "thresholds",
                rec["thresholds_sha256"],
                {
                    "period": period,
                    "baseline": baseline,
                    "field": "thresholds_sha256",
                    "scorecard": scorecard,
                },
            )
            record(
                rec["store"],
                "reference_forecast",
                rec["forecast_sha256"],
                {
                    "period": period,
                    "baseline": baseline,
                    "field": "forecast_sha256",
                    "scorecard": scorecard,
                },
            )

    if not periods:
        raise BundleError(f"No periods pinned for year {year} in {references_path.name}")

    items = [
        PinnedItem(
            relpath=relpath,
            kind=kind,
            expected_sha256=digest,
            pinned_by=tuple(
                sorted(citations, key=lambda c: (c["period"], c["baseline"], c["field"]))
            ),
        )
        for relpath, (kind, digest, citations) in sorted(seen.items())
    ]
    return inventory, periods, items


def _collect_member_files(data_dir: Path, relpath: str) -> tuple[str, int, list[dict[str, Any]]]:
    """Return ``(format, total_size_bytes, files_manifest)`` for one pinned file or Zarr store."""
    target = data_dir / relpath
    if target.is_symlink():
        raise BundleError(f"Refusing symlink: {relpath}")
    if target.is_file():
        size = target.stat().st_size
        return (
            "netcdf",
            size,
            [{"path": relpath, "size_bytes": size, "sha256": _sha256_file(target)}],
        )
    if target.is_dir():
        files: list[dict[str, Any]] = []
        total = 0
        for item in sorted(target.rglob("*")):
            if item.is_symlink():
                raise BundleError(f"Refusing symlink inside store: {item.relative_to(data_dir).as_posix()}")
            if item.is_file():
                rel = item.relative_to(data_dir).as_posix()
                _validate_relpath(rel, f"store member {rel}")
                size = item.stat().st_size
                total += size
                files.append({"path": rel, "size_bytes": size, "sha256": _sha256_file(item)})
        if not files:
            raise BundleError(f"Store directory is empty: {relpath}")
        return "zarr", total, files
    raise BundleError(f"Missing or unsupported path: {relpath}")


def _check_free_space(path: Path, min_free_gb: float, required_extra_bytes: int = 0) -> float:
    """Ensure free space on the filesystem holding ``path`` stays above ``min_free_gb``."""
    anchor = path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise BundleError(
            f"Insufficient free disk space: {free_gb:.2f} GiB available, "
            f"minimum required is {min_free_gb:.2f} GiB"
        )
    projected_gb = (usage.free - required_extra_bytes) / (1024**3)
    if required_extra_bytes > 0 and projected_gb < min_free_gb:
        raise BundleError(
            f"Projected free disk space after packing ({projected_gb:.2f} GiB) would fall below "
            f"the {min_free_gb:.2f} GiB floor"
        )
    return free_gb


def render_bundle_readme(
    *,
    year: int,
    periods: list[str],
    entries: list[dict[str, Any]],
    harness_version: dict[str, Any],
    protocol_version: str,
) -> str:
    total_bytes = sum(int(e["size_bytes"]) for e in entries)
    total_mib = total_bytes / (1024**2)
    return (
        f"# WRFComp {year} Scoring Data Bundle\n\n"
        f"This archive contains the exact verification datasets pinned by `experiments/REFERENCES.json`\n"
        f"for scoring candidate weather forecasts on **{year}** under WRFComp harness version\n"
        f"`{harness_version.get('wrfcomp', 'unknown')}` (protocol `{protocol_version}`).\n\n"
        f"## Contents\n\n"
        f"- **Periods covered ({len(periods)}):** {', '.join(periods)}\n"
        f"- **Pinned datasets:** {len(entries)} ({total_mib:.1f} MiB uncompressed)\n"
        f"  - Station ground truth (`truth/`): hourly NOAA ISD station observations for {year}\n"
        f"  - Tail thresholds (`thresholds/`): monthly station-level 95th-percentile thresholds\n"
        f"  - Reference forecasts (`gate1/`): pinned `climatology.zarr` and `gefs.zarr` stores\n"
        f"- **`MANIFEST.json`:** per-dataset harness `content_sha256`, per-file byte SHA-256, sizes, and\n"
        f"  pin provenance from `experiments/REFERENCES.json`.\n\n"
        f"This bundle contains **no** citizen weather-station data, **no** candidate models, and **no**\n"
        f"data from the sealed test split (2025 onward).\n\n"
        f"## Data Sources and License\n\n"
        f"- **Upstream sources (public domain):**\n"
        f"  - NOAA NCEI Integrated Surface Database (ISD) Global Hourly station observations\n"
        f"    (U.S. federal government work; public domain under 17 U.S.C. 105).\n"
        f"  - NOAA Global Ensemble Forecast System (GEFS v12) forecasts via NOAA Open Data\n"
        f"    Dissemination (U.S. federal government work; public domain under 17 U.S.C. 105).\n"
        f"- **License for derived files (`{RECOMMENDED_LICENSE}`):**\n"
        f"  We recommend **CC0-1.0 (Creative Commons Zero 1.0 Universal Public Domain Dedication)**\n"
        f"  for this derived bundle. The upstream NOAA datasets are U.S. federal public-domain works,\n"
        f"  and these files are deterministic numeric extractions (nearest-hour matching, station-point\n"
        f"  interpolation, and empirical quantiles). CC0 avoids attribution-stacking friction in\n"
        f"  automated verification and CI pipelines while the Zenodo DOI provides standard scholarly\n"
        f"  citation metadata. (CC-BY-4.0 is an acceptable alternative if formal attribution is desired.)\n\n"
        f"## Verification and Unpacking\n\n"
        f"Use `scripts/fetch_scoring_data.py` from the `wrfcoin/wrfcomp` repository to download,\n"
        f"verify against both `MANIFEST.json` and `experiments/REFERENCES.json`, and unpack into\n"
        f"`$WRFCOMP_DATA_DIR` (default `~/wrfcomp-data`):\n\n"
        f"```bash\n"
        f"uv run python -P scripts/fetch_scoring_data.py --url <ZENODO_RECORD_URL> --year {year}\n"
        f"```\n"
    )


def _add_bytes_member(tf: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tf.addfile(info, io.BytesIO(payload))


def _add_file_member(tf: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = source.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with source.open("rb") as fh:
        tf.addfile(info, fh)


def build_bundle(
    *,
    year: int,
    data_dir: Path,
    out_dir: Path,
    references_path: Path = DEFAULT_REFERENCES,
    config_path: Path = DEFAULT_CONFIG,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    compress_level: int = DEFAULT_COMPRESS_LEVEL,
) -> dict[str, Any]:
    """Verify all pinned artifacts for ``year`` and pack a reproducible ``.tar.gz`` archive."""
    cfg = load_config(config_path)
    _check_free_space(out_dir, min_free_gb)

    inventory, periods, items = collect_pinned_items(references_path, year, cfg)
    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()

    manifest_entries: list[dict[str, Any]] = []
    total_input_bytes = 0
    for item in items:
        target = data_dir / item.relpath
        with _open_dataset(target, f"{item.kind} {item.relpath}") as ds:
            _check_not_sealed(ds, item.kind, item.relpath, cfg)
            actual = content_sha256(ds)
        if actual != item.expected_sha256:
            raise BundleError(
                f"Content hash mismatch for {item.relpath}: "
                f"computed {actual}, pinned {item.expected_sha256}"
            )
        fmt, size_bytes, files = _collect_member_files(data_dir, item.relpath)
        total_input_bytes += size_bytes
        manifest_entries.append(
            {
                "content_sha256": actual,
                "files": files,
                "format": fmt,
                "kind": item.kind,
                "path": item.relpath,
                "pinned_by": list(item.pinned_by),
                "size_bytes": size_bytes,
            }
        )

    # Guard against dipping below min_free_gb while writing the archive.
    _check_free_space(out_dir, min_free_gb, required_extra_bytes=total_input_bytes)

    manifest = {
        "entries": manifest_entries,
        "harness_version": inventory.get("harness_version", {}),
        "license": RECOMMENDED_LICENSE,
        "manifest_version": MANIFEST_VERSION,
        "periods": periods,
        "protocol_version": inventory.get("protocol_version", ""),
        "references_sha256": _sha256_file(references_path),
        "total_size_bytes": total_input_bytes,
        "year": year,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    readme_bytes = render_bundle_readme(
        year=year,
        periods=periods,
        entries=manifest_entries,
        harness_version=manifest["harness_version"],
        protocol_version=manifest["protocol_version"],
    ).encode("utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"wrfcomp-scoring-{year}.tar.gz"
    archive_path = out_dir / archive_name
    tmp_path = out_dir / f".{archive_name}.tmp"

    try:
        with tmp_path.open("wb") as raw_fh:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_fh, compresslevel=compress_level, mtime=0
            ) as gz_fh:
                with tarfile.open(fileobj=gz_fh, mode="w", format=tarfile.PAX_FORMAT) as tf:
                    _add_bytes_member(tf, "MANIFEST.json", manifest_bytes)
                    _add_bytes_member(tf, "README.md", readme_bytes)
                    for entry in manifest_entries:
                        for fmeta in entry["files"]:
                            rel = fmeta["path"]
                            _add_file_member(tf, data_dir / rel, rel)
        tmp_path.replace(archive_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    archive_sha256 = _sha256_file(archive_path)
    archive_size = archive_path.stat().st_size

    sha_sidecar = out_dir / f"{archive_name}.sha256"
    sha_sidecar.write_text(f"{archive_sha256}  {archive_name}\n", encoding="utf-8")
    manifest_sidecar = out_dir / f"wrfcomp-scoring-{year}.MANIFEST.json"
    manifest_sidecar.write_bytes(manifest_bytes)

    return {
        "archive": archive_name,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size,
        "entries": len(manifest_entries),
        "manifest_sidecar": str(manifest_sidecar),
        "periods": periods,
        "sha256_sidecar": str(sha_sidecar),
        "uncompressed_size_bytes": total_input_bytes,
        "year": year,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a verified per-year scoring data bundle.")
    parser.add_argument("--year", type=int, default=2024, help="Calendar year to bundle (default: 2024).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="WRFComp data directory (default: $WRFCOMP_DATA_DIR or ~/wrfcomp-data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for the archive (default: <data-dir>/bundle).",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=DEFAULT_REFERENCES,
        help="Path to experiments/REFERENCES.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to configs/splits.yaml.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=f"Refuse to pack if free disk space is below this GiB floor (default: {DEFAULT_MIN_FREE_GB}).",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=DEFAULT_COMPRESS_LEVEL,
        choices=range(1, 10),
        help=f"Gzip compression level 1-9 (default: {DEFAULT_COMPRESS_LEVEL}).",
    )
    args = parser.parse_args(argv)
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    out_dir = args.out_dir if args.out_dir is not None else data_dir / "bundle"
    try:
        summary = build_bundle(
            year=args.year,
            data_dir=data_dir,
            out_dir=out_dir,
            references_path=args.references,
            config_path=args.config,
            min_free_gb=args.min_free_gb,
            compress_level=args.compress_level,
        )
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "archive": summary["archive"],
                "archive_sha256": summary["archive_sha256"],
                "archive_size_bytes": summary["archive_size_bytes"],
                "entries": summary["entries"],
                "periods": summary["periods"],
                "uncompressed_size_bytes": summary["uncompressed_size_bytes"],
                "year": summary["year"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
