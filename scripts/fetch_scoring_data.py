"""Download, verify, and unpack a WRFComp scoring data bundle from a Zenodo record URL.

No Zenodo record URL is hard-coded. Pass ``--url <ZENODO_RECORD_OR_FILE_URL>`` (or
``--archive <LOCAL_ARCHIVE>`` for offline use). Every unpacked file is verified in a private
staging directory against both the archive's ``MANIFEST.json`` (size and byte SHA-256) and the
repository's ``experiments/REFERENCES.json`` (harness ``content_sha256`` and sealed-split guard)
before anything is moved into ``--data-dir``.

Usage:
    uv run python -P scripts/fetch_scoring_data.py --url <ZENODO_RECORD_URL> --year 2024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import xarray as xr

from wrfcomp.harness.align import check_forecast_not_sealed, check_not_sealed
from wrfcomp.harness.config import HarnessConfig, load_config
from wrfcomp.harness.hashing import content_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = ROOT / "experiments" / "REFERENCES.json"
DEFAULT_CONFIG = ROOT / "configs" / "splits.yaml"
MANIFEST_VERSION = 1
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
ZENODO_RECORD_RE = re.compile(r"^/(?:records?|api/records)/(?P<id>[0-9]+)/?$")
ZENODO_DOI_RE = re.compile(r"^/10\.5281/zenodo\.(?P<id>[0-9]+)/?$")


class FetchError(RuntimeError):
    """Raised when downloading, verifying, or unpacking a scoring bundle fails."""


def default_data_dir() -> Path:
    env = os.environ.get("WRFCOMP_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "wrfcomp-data"


def _validate_relpath(relpath: str, what: str) -> Path:
    if not isinstance(relpath, str) or not relpath:
        raise FetchError(f"{what}: empty or non-string relative path")
    if relpath.startswith("/") or "\\" in relpath:
        raise FetchError(f"{what}: path must be a relative POSIX path, got {relpath!r}")
    p = Path(relpath)
    if p.is_absolute() or ".." in p.parts or "." in p.parts:
        raise FetchError(f"{what}: unsafe relative path {relpath!r}")
    lowered = {part.lower() for part in p.parts}
    forbidden = lowered & FORBIDDEN_PATH_PARTS
    if forbidden:
        raise FetchError(f"{what}: forbidden path component {sorted(forbidden)} in {relpath!r}")
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


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _open_dataset(path: Path, what: str) -> xr.Dataset:
    if not path.exists():
        raise FetchError(f"{what}: missing path {path.name}")
    if path.is_symlink():
        raise FetchError(f"{what}: symlinks are refused ({path.name})")
    try:
        if path.is_dir():
            try:
                return xr.open_zarr(path, chunks=None, consolidated=True)
            except (KeyError, ValueError, FileNotFoundError):
                return xr.open_zarr(path, chunks=None, consolidated=False)
        return xr.open_dataset(path)
    except Exception as exc:
        raise FetchError(f"{what}: failed to open {path.name}: {exc}") from exc


def _check_not_sealed(ds: xr.Dataset, kind: str, relpath: str, cfg: HarnessConfig) -> None:
    sealed_start = cfg.splits.sealed_test.start
    if kind == "truth":
        if "valid_time" not in ds.coords:
            raise FetchError(f"{relpath}: truth dataset missing valid_time coordinate")
        check_not_sealed(ds["valid_time"].values, what=f"truth {relpath}", sealed_start=sealed_start)
    elif kind == "reference_forecast":
        check_forecast_not_sealed(ds, what=f"reference {relpath}", sealed_start=sealed_start)


def expected_pins(
    references_path: Path,
    year: int,
    cfg: HarnessConfig,
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """Return ``(periods, {relpath: (kind, expected_content_sha256)})`` from ``REFERENCES.json``."""
    sealed_dt = cfg.splits.sealed_test.start_utc
    if year >= sealed_dt.year:
        raise FetchError(
            f"Refusing year {year}: sealed split starts at {sealed_dt.replace(tzinfo=None).isoformat()}"
        )
    try:
        inventory = json.loads(references_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FetchError(f"Cannot read references inventory: {exc}") from exc
    if not isinstance(inventory, dict) or not isinstance(inventory.get("periods"), dict):
        raise FetchError("Invalid REFERENCES.json structure")

    periods: list[str] = []
    pins: dict[str, tuple[str, str]] = {}

    def record(relpath: str, kind: str, digest: str) -> None:
        _validate_relpath(relpath, f"REFERENCES.json {kind}")
        if relpath in pins:
            if pins[relpath] != (kind, digest):
                raise FetchError(f"Conflicting pins in REFERENCES.json for {relpath}")
        else:
            pins[relpath] = (kind, digest)

    for period in sorted(inventory["periods"]):
        m = PERIOD_RE.fullmatch(period)
        if not m or int(m.group("year")) != year:
            continue
        periods.append(period)
        entry = inventory["periods"][period]
        refs = entry.get("references")
        if not isinstance(refs, dict) or set(refs) != set(REQUIRED_BASELINES):
            raise FetchError(
                f"Period {period} must pin exactly {', '.join(REQUIRED_BASELINES)}"
            )
        for baseline in REQUIRED_BASELINES:
            rec = refs[baseline]
            record(entry["truth"], "truth", rec["truth_sha256"])
            record(entry["thresholds"], "thresholds", rec["thresholds_sha256"])
            record(rec["store"], "reference_forecast", rec["forecast_sha256"])

    if not periods:
        raise FetchError(f"No periods pinned for year {year} in {references_path.name}")
    return periods, pins


def _check_url_scheme(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https":
        return parsed
    if parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost"):
        return parsed
    raise FetchError(
        f"Refusing URL scheme {parsed.scheme!r}: use https:// (or --archive for a local file)"
    )


def resolve_download_url(url: str, year: int, *, timeout: float = 60.0) -> tuple[str, str | None]:
    """Resolve a Zenodo record URL (or direct archive URL) to ``(file_url, expected_md5)``.

    Supports:
    - Direct download URLs (returned as-is with ``expected_md5=None``)
    - Zenodo record pages: ``https://zenodo.org/records/<id>`` or ``/record/<id>``
    - Zenodo record API URLs: ``https://zenodo.org/api/records/<id>``
    - Zenodo DOI URLs: ``https://doi.org/10.5281/zenodo.<id>``
    """
    parsed = _check_url_scheme(url)
    api_url: str | None = None
    host = (parsed.hostname or "").lower()
    if host in ("zenodo.org", "www.zenodo.org", "sandbox.zenodo.org", "127.0.0.1", "localhost"):
        m = ZENODO_RECORD_RE.fullmatch(parsed.path)
        if m:
            netloc = parsed.netloc
            api_url = f"{parsed.scheme}://{netloc}/api/records/{m.group('id')}"
    elif host in ("doi.org", "dx.doi.org"):
        m = ZENODO_DOI_RE.fullmatch(parsed.path)
        if m:
            api_url = f"https://zenodo.org/api/records/{m.group('id')}"

    if api_url is None:
        return url, None

    req = urllib.request.Request(
        api_url,
        headers={"Accept": "application/json", "User-Agent": "wrfcomp-fetch-scoring-data/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise FetchError(f"Failed to query Zenodo record metadata at {api_url}: {exc}") from exc

    files = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(files, list) or not files:
        raise FetchError(f"Zenodo record at {api_url} lists no downloadable files")

    target_prefix = f"wrfcomp-scoring-{year}.tar"
    candidates = [
        f
        for f in files
        if isinstance(f, dict)
        and str(f.get("key", "")).startswith(target_prefix)
        and not str(f.get("key", "")).endswith(".sha256")
    ]
    if not candidates:
        keys = [str(f.get("key", "")) for f in files if isinstance(f, dict)]
        raise FetchError(
            f"Zenodo record has no {target_prefix}* archive (found: {', '.join(keys)})"
        )
    chosen = sorted(candidates, key=lambda f: str(f["key"]))[0]
    links = chosen.get("links") if isinstance(chosen.get("links"), dict) else {}
    file_url = links.get("self") or links.get("content")
    if not isinstance(file_url, str) or not file_url:
        raise FetchError(f"Zenodo file entry {chosen.get('key')!r} has no download link")
    _check_url_scheme(file_url)
    checksum = chosen.get("checksum")
    expected_md5: str | None = None
    if isinstance(checksum, str) and checksum.startswith("md5:"):
        expected_md5 = checksum.split(":", 1)[1].strip().lower()
    return file_url, expected_md5


def download_file(url: str, dest: Path, *, timeout: float = 300.0) -> None:
    _check_url_scheme(url)
    req = urllib.request.Request(url, headers={"User-Agent": "wrfcomp-fetch-scoring-data/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1024 * 1024)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"Download failed for {url}: {exc}") from exc


def _safe_extract_archive(archive_path: Path, staging_dir: Path) -> set[str]:
    """Safely extract regular files from ``archive_path`` into ``staging_dir``."""
    extracted_files: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:*") as tf:
            members = tf.getmembers()
            for member in members:
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise FetchError(f"Refusing non-regular tar member: {member.name!r}")
                if not (member.isfile() or member.isdir()):
                    raise FetchError(f"Refusing unsupported tar member type: {member.name!r}")
                clean_name = member.name.rstrip("/")
                if not clean_name or clean_name == ".":
                    continue
                rel = _validate_relpath(clean_name, f"archive member {member.name!r}")
                dest = (staging_dir / rel).resolve()
                if not dest.is_relative_to(staging_dir.resolve()):
                    raise FetchError(f"Archive member escapes staging directory: {member.name!r}")
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                if rel.as_posix() in extracted_files:
                    raise FetchError(f"Duplicate archive member: {rel.as_posix()!r}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    raise FetchError(f"Cannot read archive member: {member.name!r}")
                with src, dest.open("wb") as out_fh:
                    shutil.copyfileobj(src, out_fh, length=1024 * 1024)
                extracted_files.add(rel.as_posix())
    except (tarfile.TarError, OSError) as exc:
        raise FetchError(f"Failed to read archive {archive_path.name}: {exc}") from exc
    return extracted_files


def verify_staged_bundle(
    staging_dir: Path,
    extracted_files: set[str],
    *,
    year: int,
    references_path: Path,
    cfg: HarnessConfig,
) -> dict[str, Any]:
    """Verify staged archive contents against ``MANIFEST.json`` AND ``REFERENCES.json``."""
    manifest_path = staging_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FetchError("Archive is missing MANIFEST.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FetchError(f"Cannot parse MANIFEST.json: {exc}") from exc

    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise FetchError(f"Unsupported manifest_version: {manifest.get('manifest_version')!r}")
    if manifest.get("year") != year:
        raise FetchError(f"Manifest year {manifest.get('year')!r} does not match requested year {year}")

    expected_periods, pins = expected_pins(references_path, year, cfg)
    if manifest.get("periods") != expected_periods:
        raise FetchError(
            f"Manifest periods {manifest.get('periods')!r} do not match REFERENCES.json {expected_periods!r}"
        )

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise FetchError("MANIFEST.json has no entries")

    manifest_paths: dict[str, dict[str, Any]] = {}
    allowed_files: set[str] = {"MANIFEST.json", "README.md"}

    for entry in entries:
        if not isinstance(entry, dict):
            raise FetchError("Invalid entry in MANIFEST.json")
        relpath = entry.get("path")
        _validate_relpath(str(relpath), "MANIFEST.json entry")
        if relpath in manifest_paths:
            raise FetchError(f"Duplicate entry in MANIFEST.json: {relpath}")
        manifest_paths[relpath] = entry
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise FetchError(f"MANIFEST.json entry {relpath} has no files list")
        entry_total = 0
        for fmeta in files:
            fpath = fmeta.get("path")
            _validate_relpath(str(fpath), f"MANIFEST.json file under {relpath}")
            if fpath != relpath and not fpath.startswith(f"{relpath}/"):
                raise FetchError(f"MANIFEST.json file {fpath!r} does not lie under {relpath!r}")
            staged_file = staging_dir / fpath
            if not staged_file.is_file():
                raise FetchError(f"Missing file listed in MANIFEST.json: {fpath}")
            actual_size = staged_file.stat().st_size
            if actual_size != fmeta.get("size_bytes"):
                raise FetchError(
                    f"Size mismatch for {fpath}: got {actual_size}, manifest has {fmeta.get('size_bytes')}"
                )
            actual_sha = _sha256_file(staged_file)
            if actual_sha != fmeta.get("sha256"):
                raise FetchError(
                    f"Byte SHA-256 mismatch for {fpath}: got {actual_sha}, manifest has {fmeta.get('sha256')}"
                )
            entry_total += actual_size
            allowed_files.add(fpath)
        if entry_total != entry.get("size_bytes"):
            raise FetchError(
                f"Entry total size mismatch for {relpath}: sum={entry_total}, "
                f"manifest={entry.get('size_bytes')}"
            )

    unexpected = sorted(extracted_files - allowed_files)
    if unexpected:
        raise FetchError(f"Archive contains unmanifested extra files: {unexpected[:5]}")

    missing_from_manifest = sorted(set(pins) - set(manifest_paths))
    extra_in_manifest = sorted(set(manifest_paths) - set(pins))
    if missing_from_manifest:
        raise FetchError(f"Archive is missing pinned artifacts: {missing_from_manifest}")
    if extra_in_manifest:
        raise FetchError(f"Archive contains unpinned artifacts: {extra_in_manifest}")

    # Verify dataset-level content_sha256 against BOTH MANIFEST.json and REFERENCES.json
    for relpath, (kind, ref_sha256) in sorted(pins.items()):
        entry = manifest_paths[relpath]
        if entry.get("kind") != kind:
            raise FetchError(f"Kind mismatch for {relpath}: manifest={entry.get('kind')}, expected={kind}")
        if entry.get("content_sha256") != ref_sha256:
            raise FetchError(
                f"MANIFEST.json content_sha256 disagrees with REFERENCES.json for {relpath}: "
                f"manifest={entry.get('content_sha256')}, pinned={ref_sha256}"
            )
        staged_target = staging_dir / relpath
        with _open_dataset(staged_target, f"{kind} {relpath}") as ds:
            _check_not_sealed(ds, kind, relpath, cfg)
            actual_content = content_sha256(ds)
        if actual_content != ref_sha256:
            raise FetchError(
                f"Dataset content_sha256 mismatch for {relpath}: "
                f"computed={actual_content}, pinned={ref_sha256}"
            )

    return manifest


def _publish_staged(
    staging_dir: Path,
    data_dir: Path,
    entries: list[dict[str, Any]],
    *,
    overwrite: bool,
    cfg: HarnessConfig,
) -> list[str]:
    """Atomically install verified entries from ``staging_dir`` into ``data_dir``."""
    # Pre-check every destination before modifying data_dir so failures leave data_dir untouched.
    actions: list[tuple[str, str]] = []  # (relpath, "install" | "replace" | "keep")
    for entry in entries:
        relpath = entry["path"]
        dest = data_dir / relpath
        if not dest.exists():
            actions.append((relpath, "install"))
            continue
        if overwrite:
            actions.append((relpath, "replace"))
            continue
        # If target already exists, allow it only if it already has the exact pinned content hash.
        with _open_dataset(dest, f"existing {relpath}") as ds:
            _check_not_sealed(ds, entry["kind"], relpath, cfg)
            existing_hash = content_sha256(ds)
        if existing_hash != entry["content_sha256"]:
            raise FetchError(
                f"Destination {relpath} already exists with different content "
                f"(found {existing_hash}, expected {entry['content_sha256']}); "
                "re-run with --overwrite to replace it"
            )
        actions.append((relpath, "keep"))

    installed: list[str] = []
    for relpath, action in actions:
        if action == "keep":
            continue
        src = staging_dir / relpath
        dest = data_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if action == "replace":
            backup = dest.parent / f".{dest.name}.old.{os.getpid()}"
            if backup.exists():
                shutil.rmtree(backup) if backup.is_dir() else backup.unlink()
            os.replace(dest, backup)
            try:
                os.replace(src, dest)
            except Exception:
                os.replace(backup, dest)
                raise
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        else:
            os.replace(src, dest)
        installed.append(relpath)
    return installed


def fetch_scoring_data(
    *,
    year: int,
    data_dir: Path,
    url: str | None = None,
    archive: Path | None = None,
    references_path: Path = DEFAULT_REFERENCES,
    config_path: Path = DEFAULT_CONFIG,
    expected_sha256: str | None = None,
    overwrite: bool = False,
    keep_archive: bool = False,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Download (or open local), verify, and unpack a scoring data bundle into ``data_dir``."""
    if (url is None) == (archive is None):
        raise FetchError("Specify exactly one of --url or --archive")

    cfg = load_config(config_path)
    # Fail fast if the year is sealed or unpinned before downloading anything.
    expected_pins(references_path, year, cfg)

    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".fetch-scoring-", dir=data_dir) as tmp_str:
        tmp_dir = Path(tmp_str)
        staging_dir = tmp_dir / "stage"
        staging_dir.mkdir()

        expected_md5: str | None = None
        if url is not None:
            file_url, expected_md5 = resolve_download_url(url, year, timeout=min(timeout, 60.0))
            archive_path = tmp_dir / f"wrfcomp-scoring-{year}.tar.gz"
            download_file(file_url, archive_path, timeout=timeout)
        else:
            assert archive is not None
            archive_path = archive.resolve()
            if not archive_path.is_file():
                raise FetchError(f"Archive not found: {archive_path.name}")

        archive_sha256 = _sha256_file(archive_path)
        if expected_sha256 is not None and archive_sha256 != expected_sha256.strip().lower():
            raise FetchError(
                f"Archive SHA-256 mismatch: got {archive_sha256}, expected {expected_sha256.strip().lower()}"
            )
        if expected_md5 is not None:
            actual_md5 = _md5_file(archive_path)
            if actual_md5 != expected_md5:
                raise FetchError(
                    f"Archive MD5 mismatch vs Zenodo record metadata: "
                    f"got {actual_md5}, expected {expected_md5}"
                )

        extracted_files = _safe_extract_archive(archive_path, staging_dir)
        manifest = verify_staged_bundle(
            staging_dir,
            extracted_files,
            year=year,
            references_path=references_path,
            cfg=cfg,
        )
        installed = _publish_staged(
            staging_dir,
            data_dir,
            manifest["entries"],
            overwrite=overwrite,
            cfg=cfg,
        )

        if keep_archive and url is not None:
            bundle_dir = data_dir / "bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive_path, bundle_dir / f"wrfcomp-scoring-{year}.tar.gz")

    return {
        "archive_sha256": archive_sha256,
        "entries_verified": len(manifest["entries"]),
        "installed": installed,
        "periods": manifest["periods"],
        "year": year,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download, verify, and unpack a WRFComp scoring data bundle."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--url",
        type=str,
        default=None,
        help="Zenodo record URL (or direct archive URL) for the scoring bundle.",
    )
    source.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Local bundle archive path (offline mode; no network access).",
    )
    parser.add_argument("--year", type=int, default=2024, help="Scoring bundle year (default: 2024).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Target WRFComp data directory (default: $WRFCOMP_DATA_DIR or ~/wrfcomp-data).",
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
        "--expected-sha256",
        type=str,
        default=None,
        help="Optional expected SHA-256 hex digest of the downloaded archive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files/stores in --data-dir even if their hashes differ.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep a copy of the downloaded archive under <data-dir>/bundle/.",
    )
    args = parser.parse_args(argv)
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    try:
        summary = fetch_scoring_data(
            year=args.year,
            data_dir=data_dir,
            url=args.url,
            archive=args.archive,
            references_path=args.references,
            config_path=args.config,
            expected_sha256=args.expected_sha256,
            overwrite=args.overwrite,
            keep_archive=args.keep_archive,
        )
    except FetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
