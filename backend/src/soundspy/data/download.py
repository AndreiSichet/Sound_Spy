"""Download, verify and extract the training datasets into ``paths.external`` (drive D:).

One folder per dataset, ``<external>/<dataset>/``:

* ``archives/``  downloaded files, kept until ``soundspy data cleanup-archives``;
* the extracted data (``ontology.json``, ``ESC-50-master/``, ``FSD50K.dev_audio/`` ...);
* ``SOURCE.json`` where it came from: URLs, date, version, checksums, license;
* ``.done``       written last, after the extracted files were counted against the CSVs.
  Re-running a download whose ``.done`` exists does nothing (idempotent).

Downloads are resumable: bytes go to ``<name>.part`` and an HTTP ``Range`` request continues
where the previous attempt stopped. A file gets its final name only after its checksum passed,
so an existing final file is a verified file. Network errors are retried with exponential
backoff. FSD50K's audio is a split zip (``.zip`` + ``.z01`` ...) that only 7-Zip extracts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

USER_AGENT = "soundspy-data/0.1 (+research dataset download)"
CHUNK_BYTES = 1 << 20
GB = 1_000_000_000

DATASETS = ("ontology", "esc50", "fsd50k")

ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
ONTOLOGY_REPO = "audioset/ontology"
ONTOLOGY_LICENSE = "CC BY-SA 4.0 (AudioSet ontology, Google)"

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
ESC50_REPO = "karolpiczak/ESC-50"
ESC50_ARCHIVE = "ESC-50-master.zip"
ESC50_LICENSE = (
    "CC BY-NC 3.0 (ESC-50 compilation by K. J. Piczak; clips from freesound.org, "
    "see meta/esc50.csv for each clip's freesound id)"
)
ESC50_CLIPS = 2000

FSD50K_RECORD = "4060432"
FSD50K_API = f"https://zenodo.org/api/records/{FSD50K_RECORD}"
FSD50K_LICENSE = (
    "Dataset: CC BY 4.0. Every clip keeps its own Creative Commons license "
    "(CC0, CC-BY, CC-BY-NC, Sampling+), listed per clip in "
    "FSD50K.metadata/*_clips_info_FSD50K.json."
)
FSD50K_EXTRACTED_BYTES = 36 * GB
"""Extracted FSD50K (audio + metadata) measured ~34 GB; a little margin on top."""

SEVEN_ZIP_HINT = "Install 7-Zip: `winget install 7zip.7zip`"

# Margin kept free on the drive on top of every estimate.
FREE_SPACE_MARGIN = 2 * GB


class DownloadError(RuntimeError):
    """A download, checksum or extraction step failed (message is shown to the user)."""


class ChecksumError(DownloadError):
    pass


class InsufficientSpaceError(DownloadError):
    pass


class SevenZipMissingError(DownloadError):
    pass


@dataclass(frozen=True)
class RemoteFile:
    """One file to download. ``md5`` (when known) is checked; SHA-256 is always recorded."""

    name: str
    url: str
    size: int | None = None
    md5: str | None = None


@dataclass
class FileRecord:
    name: str
    url: str
    size: int
    md5: str
    sha256: str
    md5_verified: bool


@dataclass
class DatasetResult:
    dataset: str
    status: str  # "done" | "already done"
    root: Path
    seconds: float = 0.0
    files: list[FileRecord] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Disk space


def free_bytes(path: Path) -> int:
    """Free bytes on the drive holding ``path`` (works before the folder exists)."""
    probe = Path(path).resolve()
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    return shutil.disk_usage(probe).free


def check_free_space(
    path: Path, needed: int, what: str, *, free: Callable[[Path], int] = free_bytes
) -> None:
    available = free(path)
    if available < needed + FREE_SPACE_MARGIN:
        raise InsufficientSpaceError(
            f"{what} needs about {needed / GB:.1f} GB free on the drive of {path} "
            f"(+{FREE_SPACE_MARGIN / GB:.0f} GB margin), but only {available / GB:.1f} GB are "
            "free. Free some space or point paths.external to another drive in default.yaml."
        )


# --------------------------------------------------------------------------------------------
# HTTP


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def hash_file(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(8 * CHUNK_BYTES):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def _progress(total: int | None, initial: int, name: str, enabled: bool):
    from tqdm import tqdm

    return tqdm(
        total=total,
        initial=initial,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=name,
        disable=not enabled,
        mininterval=1.0,
        file=sys.stdout,
    )


def download_file(
    session: requests.Session,
    remote: RemoteFile,
    dest_dir: Path,
    *,
    retries: int = 8,
    backoff_s: float = 2.0,
    max_backoff_s: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    progress: bool = True,
    timeout: tuple[float, float] = (15.0, 60.0),
) -> FileRecord:
    """Download ``remote`` into ``dest_dir`` with resume, retries and checksum verification.

    An existing ``dest_dir/<name>`` is trusted (it was renamed from ``.part`` only after
    passing its checksum); its hashes are recomputed for the record.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / remote.name
    part = dest_dir / (remote.name + ".part")
    if final.exists() and (remote.size is None or final.stat().st_size == remote.size):
        md5, sha = hash_file(final)
        _check_md5(remote, md5, final)
        return FileRecord(remote.name, remote.url, final.stat().st_size, md5, sha,
                          remote.md5 is not None)

    attempt = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if remote.size is not None and have > remote.size:
            log.warning("%s: .part is larger than expected, restarting", remote.name)
            part.unlink()
            have = 0
        if remote.size is not None and have == remote.size:
            break
        try:
            if _fetch(session, remote, part, have, progress=progress, timeout=timeout):
                break
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError, _RetryableHTTP) as exc:
            attempt += 1
            if attempt > retries:
                raise DownloadError(
                    f"{remote.name}: giving up after {retries} retries ({exc}). Run the same "
                    "command again later: the download resumes where it stopped."
                ) from exc
            wait = min(max_backoff_s, backoff_s * 2 ** (attempt - 1))
            log.warning("%s: %s; retry %d/%d in %.0f s", remote.name, exc, attempt, retries, wait)
            print(f"  {remote.name}: {exc}; retry {attempt}/{retries} in {wait:.0f} s",
                  flush=True)
            sleep(wait)

    md5, sha = hash_file(part)
    try:
        _check_md5(remote, md5, part)
    except ChecksumError:
        part.unlink(missing_ok=True)
        raise
    size = part.stat().st_size
    os.replace(part, final)
    return FileRecord(remote.name, remote.url, size, md5, sha, remote.md5 is not None)


class _RetryableHTTP(Exception):
    pass


def _check_md5(remote: RemoteFile, md5: str, path: Path) -> None:
    if remote.md5 is not None and md5.lower() != remote.md5.lower():
        raise ChecksumError(
            f"{remote.name}: MD5 mismatch (expected {remote.md5}, got {md5}). The corrupt file "
            f"was deleted; run the download again. ({path})"
        )


def _fetch(
    session: requests.Session,
    remote: RemoteFile,
    part: Path,
    have: int,
    *,
    progress: bool,
    timeout: tuple[float, float],
) -> bool:
    """One HTTP attempt appending to ``part``. True when the file is complete."""
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(remote.url, headers=headers, stream=True, timeout=timeout) as resp:
        if resp.status_code == 416 and have:
            # Range not satisfiable: we already have everything (size was unknown).
            return True
        if resp.status_code in (429, 500, 502, 503, 504):
            raise _RetryableHTTP(f"HTTP {resp.status_code}")
        if resp.status_code not in (200, 206):
            raise DownloadError(f"{remote.name}: HTTP {resp.status_code} from {remote.url}")
        mode = "ab"
        if have and resp.status_code == 200:
            log.info("%s: server ignored the Range request, restarting", remote.name)
            have, mode = 0, "wb"
        length = resp.headers.get("Content-Length")
        total = remote.size if remote.size is not None else (
            have + int(length) if length is not None else None
        )
        received = 0
        with open(part, mode) as fh, _progress(total, have, remote.name, progress) as bar:
            for block in resp.iter_content(chunk_size=CHUNK_BYTES):
                if block:
                    fh.write(block)
                    received += len(block)
                    bar.update(len(block))
    done = have + received
    if remote.size is not None:
        if done < remote.size:
            raise _RetryableHTTP(f"connection closed at {done}/{remote.size} bytes")
        return True
    if length is not None and received < int(length):
        raise _RetryableHTTP(f"connection closed after {received}/{length} bytes")
    return True


def github_head_commit(
    session: requests.Session, repo: str, api: str = "https://api.github.com"
) -> str | None:
    """Commit SHA of ``master`` (best effort; recorded as the dataset version)."""
    try:
        resp = session.get(f"{api}/repos/{repo}/commits/master", timeout=15)
        if resp.ok:
            return str(resp.json()["sha"])
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


# --------------------------------------------------------------------------------------------
# Extraction


def find_7zip() -> Path | None:
    for name in ("7z", "7z.exe", "7za"):
        found = shutil.which(name)
        if found:
            return Path(found)
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 r"C:\Program Files", r"C:\Program Files (x86)"):
        if base:
            candidate = Path(base) / "7-Zip" / "7z.exe"
            if candidate.exists():
                return candidate
    return None


def extract_zip(archive: Path, dest: Path) -> int:
    """Extract a normal zip with Python, refusing entries that escape ``dest``. Returns count."""
    dest = dest.resolve()
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if not target.is_relative_to(dest):
                raise DownloadError(f"{archive.name}: unsafe path in archive: {info.filename}")
            zf.extract(info, dest)
            count += 1
    return count


def extract_with_7zip(archive: Path, dest: Path, seven_zip: Path) -> None:
    """Extract (split) zips with 7-Zip: give it the ``.zip``, it finds ``.z01`` ... itself."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [str(seven_zip), "x", "-y", "-bso0", "-bsp1", f"-o{dest}", str(archive)]
    print(f"  extracting {archive.name} with 7-Zip ...", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise DownloadError(f"7-Zip failed on {archive.name} (exit code {result.returncode})")


# --------------------------------------------------------------------------------------------
# Datasets


@dataclass
class DownloadContext:
    external: Path
    session: requests.Session = field(default_factory=make_session)
    progress: bool = True
    confirm: Callable[[str], bool] = lambda _msg: True
    free: Callable[[Path], int] = free_bytes
    sleep: Callable[[float], None] = time.sleep
    seven_zip: Callable[[], Path | None] = find_7zip
    fsd50k_api: str = FSD50K_API
    github_api: str = "https://api.github.com"
    ontology_url: str = ONTOLOGY_URL
    esc50_url: str = ESC50_URL


def dataset_dir(external: Path, dataset: str) -> Path:
    return Path(external) / dataset


def is_done(external: Path, dataset: str) -> bool:
    return (dataset_dir(external, dataset) / ".done").exists()


def _write_source(root: Path, payload: dict[str, Any]) -> None:
    (root / "SOURCE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _mark_done(root: Path, summary: dict[str, Any]) -> None:
    (root / ".done").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def download(datasets: Iterable[str], ctx: DownloadContext) -> list[DatasetResult]:
    """Download (and extract) each dataset in order; ``all`` expands to every dataset."""
    names: list[str] = []
    for name in datasets:
        for item in DATASETS if name == "all" else (name,):
            if item not in DATASETS:
                raise DownloadError(f"unknown dataset {item!r} (choose from {DATASETS} or all)")
            if item not in names:
                names.append(item)
    handlers = {"ontology": download_ontology, "esc50": download_esc50,
                "fsd50k": download_fsd50k}
    return [handlers[name](ctx) for name in names]


def download_ontology(ctx: DownloadContext) -> DatasetResult:
    root = dataset_dir(ctx.external, "ontology")
    if is_done(ctx.external, "ontology"):
        return DatasetResult("ontology", "already done", root)
    start = time.monotonic()
    check_free_space(ctx.external, 50_000_000, "The AudioSet ontology", free=ctx.free)
    record = download_file(ctx.session, RemoteFile("ontology.json", ctx.ontology_url), root,
                           sleep=ctx.sleep, progress=ctx.progress)
    try:
        nodes = json.loads((root / "ontology.json").read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DownloadError(f"ontology.json is not valid JSON: {exc}") from exc
    if not isinstance(nodes, list) or not nodes or "id" not in nodes[0]:
        raise DownloadError("ontology.json does not look like the AudioSet ontology")
    _write_source(root, {
        "dataset": "AudioSet ontology",
        "url": ctx.ontology_url,
        "downloaded_at": _now(),
        "version": github_head_commit(ctx.session, ONTOLOGY_REPO, ctx.github_api) or "master",
        "license": ONTOLOGY_LICENSE,
        "files": [asdict(record)],
    })
    _mark_done(root, {"nodes": len(nodes), "finished_at": _now()})
    return DatasetResult("ontology", "done", root, time.monotonic() - start, [record])


def count_esc50(root: Path) -> tuple[int, int]:
    """(rows in meta/esc50.csv, WAV files in audio/)."""
    base = root / "ESC-50-master"
    csv_path = base / "meta" / "esc50.csv"
    if not csv_path.exists():
        raise DownloadError(f"ESC-50: {csv_path} missing after extraction")
    with open(csv_path, encoding="utf-8") as fh:
        rows = sum(1 for line in fh if line.strip()) - 1
    wavs = sum(1 for _ in (base / "audio").glob("*.wav"))
    return rows, wavs


def download_esc50(ctx: DownloadContext) -> DatasetResult:
    root = dataset_dir(ctx.external, "esc50")
    if is_done(ctx.external, "esc50"):
        return DatasetResult("esc50", "already done", root)
    start = time.monotonic()
    archives = root / "archives"
    have_archive = (archives / ESC50_ARCHIVE).exists()
    check_free_space(ctx.external, (0 if have_archive else 1 * GB) + 1 * GB, "ESC-50",
                     free=ctx.free)
    record = download_file(ctx.session, RemoteFile(ESC50_ARCHIVE, ctx.esc50_url), archives,
                           sleep=ctx.sleep, progress=ctx.progress)
    print("  extracting ESC-50 ...", flush=True)
    extract_zip(archives / ESC50_ARCHIVE, root)
    rows, wavs = count_esc50(root)
    if rows != ESC50_CLIPS or wavs != rows:
        raise DownloadError(f"ESC-50: expected {ESC50_CLIPS} clips, CSV has {rows}, "
                            f"audio/ has {wavs} WAV files")
    _write_source(root, {
        "dataset": "ESC-50",
        "url": ctx.esc50_url,
        "downloaded_at": _now(),
        "version": github_head_commit(ctx.session, ESC50_REPO, ctx.github_api) or "master",
        "license": ESC50_LICENSE,
        "files": [asdict(record)],
    })
    _mark_done(root, {"clips": wavs, "finished_at": _now()})
    return DatasetResult("esc50", "done", root, time.monotonic() - start, [record])


def fsd50k_files(ctx: DownloadContext) -> tuple[list[RemoteFile], dict[str, Any]]:
    """File list (name, URL, size, MD5) from the Zenodo record API, never hard-coded."""
    try:
        resp = ctx.session.get(ctx.fsd50k_api, timeout=30)
        resp.raise_for_status()
        record = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise DownloadError(f"cannot read the Zenodo record {ctx.fsd50k_api}: {exc}") from exc
    files = []
    for item in record.get("files", []):
        checksum = str(item.get("checksum", ""))
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        url = item.get("links", {}).get("self") or (
            f"{ctx.fsd50k_api}/files/{item['key']}/content"
        )
        files.append(RemoteFile(item["key"], url, int(item["size"]), md5))
    if not files:
        raise DownloadError("the Zenodo record lists no files")
    return sorted(files, key=lambda f: f.name), record


def _count_csv_rows(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip()) - 1


def count_fsd50k(root: Path) -> dict[str, tuple[int, int]]:
    """``{"dev": (csv rows, wav files), "eval": (...)}``."""
    truth = root / "FSD50K.ground_truth"
    out = {}
    for part in ("dev", "eval"):
        csv_path = truth / f"{part}.csv"
        if not csv_path.exists():
            raise DownloadError(f"FSD50K: {csv_path} missing after extraction")
        wavs = sum(1 for _ in (root / f"FSD50K.{part}_audio").glob("*.wav"))
        out[part] = (_count_csv_rows(csv_path), wavs)
    return out


def download_fsd50k(ctx: DownloadContext) -> DatasetResult:
    root = dataset_dir(ctx.external, "fsd50k")
    if is_done(ctx.external, "fsd50k"):
        return DatasetResult("fsd50k", "already done", root)
    start = time.monotonic()
    archives = root / "archives"
    remotes, record = fsd50k_files(ctx)
    remaining = 0
    for remote in remotes:
        final, part = archives / remote.name, archives / (remote.name + ".part")
        if final.exists():
            continue
        have = part.stat().st_size if part.exists() else 0
        remaining += max(0, (remote.size or 0) - have)
    total = sum(r.size or 0 for r in remotes)
    needed = remaining + FSD50K_EXTRACTED_BYTES
    check_free_space(ctx.external, needed, "FSD50K (download + extraction)", free=ctx.free)
    seven_zip = ctx.seven_zip()
    print(f"FSD50K: {len(remotes)} files, {total / GB:.1f} GB in total, "
          f"{remaining / GB:.1f} GB still to download; extracted ~{FSD50K_EXTRACTED_BYTES / GB:.0f}"
          " GB.", flush=True)
    if seven_zip is None:
        print(f"  Note: extracting the split audio zips needs 7-Zip, which was not found. "
              f"Downloading anyway. {SEVEN_ZIP_HINT}", flush=True)
    if remaining and not ctx.confirm(f"Download {remaining / GB:.1f} GB to {archives}?"):
        raise DownloadError("cancelled")

    records = [download_file(ctx.session, remote, archives, sleep=ctx.sleep,
                             progress=ctx.progress) for remote in remotes]

    # Plain zips (ground truth, metadata, doc) with Python; split audio zips need 7-Zip.
    names = {r.name for r in remotes}
    split_zips = sorted(n for n in names if n.endswith(".zip")
                        and n.replace(".zip", ".z01") in names)
    plain_zips = sorted(n for n in names if n.endswith(".zip") and n not in split_zips)
    for name in plain_zips:
        marker = archives / f"{name}.extracted"
        if not marker.exists():
            print(f"  extracting {name} ...", flush=True)
            extract_zip(archives / name, root)
            marker.write_text(_now(), encoding="utf-8")
    if split_zips:
        seven_zip = ctx.seven_zip()
        if seven_zip is None:
            raise SevenZipMissingError(
                f"FSD50K is downloaded and verified, but its audio is a split zip. "
                f"{SEVEN_ZIP_HINT}, then run `soundspy data download fsd50k` again "
                "(nothing is downloaded twice)."
            )
        for name in split_zips:
            marker = archives / f"{name}.extracted"
            if not marker.exists():
                extract_with_7zip(archives / name, root, seven_zip)
                marker.write_text(_now(), encoding="utf-8")

    counts = count_fsd50k(root)
    for part, (rows, wavs) in counts.items():
        if rows != wavs:
            raise DownloadError(f"FSD50K {part}: {part}.csv lists {rows} clips but "
                                f"FSD50K.{part}_audio has {wavs} WAV files")
    meta = record.get("metadata", {})
    license_info = meta.get("license")
    _write_source(root, {
        "dataset": "FSD50K",
        "url": f"https://zenodo.org/records/{FSD50K_RECORD}",
        "api": ctx.fsd50k_api,
        "doi": record.get("doi") or meta.get("doi"),
        "downloaded_at": _now(),
        "version": meta.get("version"),
        "license": FSD50K_LICENSE,
        "zenodo_license": license_info,
        "files": [asdict(r) for r in records],
    })
    _mark_done(root, {"clips": {k: v[1] for k, v in counts.items()}, "finished_at": _now()})
    return DatasetResult("fsd50k", "done", root, time.monotonic() - start, records)


# --------------------------------------------------------------------------------------------
# Cleanup


@dataclass(frozen=True)
class ArchiveDir:
    dataset: str
    path: Path
    size: int
    files: int


def archive_dirs(external: Path) -> list[ArchiveDir]:
    """``archives/`` folders of datasets whose extraction finished (``.done`` present)."""
    found = []
    for dataset in DATASETS:
        folder = dataset_dir(external, dataset) / "archives"
        if not folder.is_dir() or not is_done(external, dataset):
            continue
        files = [p for p in folder.iterdir() if p.is_file()]
        found.append(ArchiveDir(dataset, folder, sum(p.stat().st_size for p in files),
                                len(files)))
    return found


def cleanup_archives(dirs: Iterable[ArchiveDir]) -> int:
    freed = 0
    for item in dirs:
        freed += item.size
        shutil.rmtree(item.path)
    return freed
