"""Downloader against a local HTTP server: resume, checksums, free space, .done, extraction."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from soundspy.data import download as dl

MiB = 1 << 20


@dataclass
class FakeServer:
    files: dict[str, bytes] = field(default_factory=dict)
    cut_once: dict[str, int] = field(default_factory=dict)  # path -> bytes sent before dropping
    ignore_range: set[str] = field(default_factory=set)
    requests: list[tuple[str, str | None]] = field(default_factory=list)
    base: str = ""


def _handler(server: FakeServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # silence
            pass

        def do_GET(self) -> None:  # noqa: N802
            rng = self.headers.get("Range")
            server.requests.append((self.path, rng))
            body = server.files.get(self.path)
            if body is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start = 0
            if rng and self.path not in server.ignore_range:
                start = int(rng.split("=")[1].split("-")[0])
                if start >= len(body):
                    self.send_response(416)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            else:
                self.send_response(200)
            payload = body[start:]
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            cut = server.cut_once.pop(self.path, None)
            if cut is not None:
                self.wfile.write(payload[:cut])
                self.wfile.flush()
                self.close_connection = True
                self.connection.shutdown(2)
                return
            self.wfile.write(payload)

    return Handler


@pytest.fixture
def server() -> Iterator[FakeServer]:
    state = FakeServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    state.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield state
    httpd.shutdown()
    httpd.server_close()


def payload(n: int, seed: int = 1) -> bytes:
    return bytes((i * 31 + seed) % 251 for i in range(n))


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def ctx_for(server: FakeServer, external: Path, **kw) -> dl.DownloadContext:
    defaults = dict(
        external=external, progress=False, sleep=lambda s: None,
        free=lambda p: 10**15, seven_zip=lambda: None,
        fsd50k_api=f"{server.base}/api/records/1", github_api=f"{server.base}/gh",
        ontology_url=f"{server.base}/ontology.json", esc50_url=f"{server.base}/esc50.zip",
    )
    defaults.update(kw)
    return dl.DownloadContext(**defaults)


def test_resume_after_interruption(server: FakeServer, tmp_path: Path) -> None:
    data = payload(4 * MiB)
    server.files["/big.bin"] = data
    server.cut_once["/big.bin"] = int(2.5 * MiB)
    waits = []
    remote = dl.RemoteFile("big.bin", f"{server.base}/big.bin", len(data), md5(data))
    record = dl.download_file(dl.make_session(), remote, tmp_path, sleep=waits.append,
                              progress=False)
    assert (tmp_path / "big.bin").read_bytes() == data
    assert not (tmp_path / "big.bin.part").exists()
    assert record.md5 == md5(data) and record.md5_verified
    assert record.sha256 == hashlib.sha256(data).hexdigest()
    ranges = [r for p, r in server.requests if p == "/big.bin"]
    assert ranges[0] is None
    assert ranges[1] is not None and int(ranges[1].split("=")[1].rstrip("-")) >= 2 * MiB
    assert waits == [2.0]


def test_resume_from_existing_part_file(server: FakeServer, tmp_path: Path) -> None:
    data = payload(3 * MiB)
    server.files["/f.bin"] = data
    (tmp_path / "f.bin.part").write_bytes(data[:MiB])
    remote = dl.RemoteFile("f.bin", f"{server.base}/f.bin", len(data), md5(data))
    dl.download_file(dl.make_session(), remote, tmp_path, progress=False)
    assert (tmp_path / "f.bin").read_bytes() == data
    assert server.requests == [("/f.bin", f"bytes={MiB}-")]


def test_server_ignoring_range_restarts_cleanly(server: FakeServer, tmp_path: Path) -> None:
    data = payload(2 * MiB)
    server.files["/f.bin"] = data
    server.ignore_range.add("/f.bin")
    (tmp_path / "f.bin.part").write_bytes(b"garbage" * 100)
    remote = dl.RemoteFile("f.bin", f"{server.base}/f.bin", len(data), md5(data))
    dl.download_file(dl.make_session(), remote, tmp_path, progress=False)
    assert (tmp_path / "f.bin").read_bytes() == data


def test_checksum_mismatch_is_an_error(server: FakeServer, tmp_path: Path) -> None:
    server.files["/f.bin"] = payload(1000)
    remote = dl.RemoteFile("f.bin", f"{server.base}/f.bin", 1000, "0" * 32)
    with pytest.raises(dl.ChecksumError, match="MD5 mismatch"):
        dl.download_file(dl.make_session(), remote, tmp_path, progress=False)
    assert not (tmp_path / "f.bin").exists()
    assert not (tmp_path / "f.bin.part").exists()


def test_gives_up_after_retries_with_resume_hint(tmp_path: Path) -> None:
    remote = dl.RemoteFile("f.bin", "http://127.0.0.1:9/f.bin", 10)
    with pytest.raises(dl.DownloadError, match="resumes"):
        dl.download_file(dl.make_session(), remote, tmp_path, retries=2, sleep=lambda s: None,
                         progress=False, timeout=(0.5, 0.5))


def test_free_space_check_aborts_before_downloading(server: FakeServer, tmp_path: Path) -> None:
    server.files["/ontology.json"] = b"[]"
    ctx = ctx_for(server, tmp_path / "ext", free=lambda p: 10_000_000)
    with pytest.raises(dl.InsufficientSpaceError, match="GB free"):
        dl.download(["ontology"], ctx)
    assert server.requests == []


def test_ontology_done_marker_makes_download_idempotent(server: FakeServer,
                                                        tmp_path: Path) -> None:
    server.files["/ontology.json"] = json.dumps(
        [{"id": "/m/a", "name": "A", "child_ids": []}]).encode()
    ctx = ctx_for(server, tmp_path / "ext")
    first = dl.download(["ontology"], ctx)
    assert first[0].status == "done"
    root = tmp_path / "ext" / "ontology"
    source = json.loads((root / "SOURCE.json").read_text())
    expected = hashlib.sha256(server.files["/ontology.json"]).hexdigest()
    assert source["files"][0]["sha256"] == expected
    assert source["version"] == "master"  # GitHub API unreachable in tests -> fallback
    n = len(server.requests)
    second = dl.download(["ontology", "ontology"], ctx)
    assert [r.status for r in second] == ["already done"]
    assert len(server.requests) == n


def make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_esc50_download_extracts_and_counts(server: FakeServer, tmp_path: Path,
                                           monkeypatch) -> None:
    monkeypatch.setattr(dl, "ESC50_CLIPS", 2)
    csv = b"filename,fold,target,category,esc10,src_file,take\n1-1-A-0.wav,1,0,dog,True,1,A\n" \
          b"1-2-A-0.wav,1,0,dog,True,2,A\n"
    server.files["/esc50.zip"] = make_zip({
        "ESC-50-master/meta/esc50.csv": csv,
        "ESC-50-master/audio/1-1-A-0.wav": b"RIFF",
        "ESC-50-master/audio/1-2-A-0.wav": b"RIFF",
    })
    res = dl.download(["esc50"], ctx_for(server, tmp_path / "ext"))[0]
    assert res.status == "done"
    root = tmp_path / "ext" / "esc50"
    assert (root / ".done").exists()
    assert (root / "archives" / "ESC-50-master.zip").exists()  # kept until cleanup
    assert (root / "ESC-50-master" / "audio" / "1-2-A-0.wav").exists()
    assert json.loads((root / "SOURCE.json").read_text())["license"].startswith("CC BY-NC")


def test_esc50_count_mismatch_is_an_error(server: FakeServer, tmp_path: Path) -> None:
    server.files["/esc50.zip"] = make_zip({
        "ESC-50-master/meta/esc50.csv": b"filename\n1-1-A-0.wav\n",
        "ESC-50-master/audio/1-1-A-0.wav": b"RIFF",
    })
    with pytest.raises(dl.DownloadError, match="expected 2000"):
        dl.download(["esc50"], ctx_for(server, tmp_path / "ext"))
    assert not (tmp_path / "ext" / "esc50" / ".done").exists()


def test_unsafe_zip_entries_are_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(make_zip({"../escape.txt": b"x"}))
    with pytest.raises(dl.DownloadError, match="unsafe"):
        dl.extract_zip(archive, tmp_path / "out")


def zenodo(server: FakeServer, files: dict[str, bytes]) -> None:
    record = {"doi": "10.5281/zenodo.1", "metadata": {"version": "1.0", "license": {"id": "x"}},
              "files": []}
    for name, data in files.items():
        server.files[f"/files/{name}"] = data
        record["files"].append({"key": name, "size": len(data), "checksum": f"md5:{md5(data)}",
                                "links": {"self": f"{server.base}/files/{name}"}})
    server.files["/api/records/1"] = json.dumps(record).encode()


def fsd_archives() -> dict[str, bytes]:
    return {
        "FSD50K.ground_truth.zip": make_zip({
            "FSD50K.ground_truth/dev.csv": b"fname,labels,mids,split\n1,Dog,/m/dog,train\n",
            "FSD50K.ground_truth/eval.csv": b"fname,labels,mids\n2,Dog,/m/dog\n",
        }),
        "FSD50K.metadata.zip": make_zip({"FSD50K.metadata/dev_clips_info_FSD50K.json": b"{}"}),
        "FSD50K.dev_audio.zip": make_zip({"FSD50K.dev_audio/1.wav": b"RIFF"}),
        "FSD50K.eval_audio.zip": make_zip({"FSD50K.eval_audio/2.wav": b"RIFF"}),
    }


def test_fsd50k_from_zenodo_record(server: FakeServer, tmp_path: Path) -> None:
    zenodo(server, fsd_archives())
    res = dl.download(["fsd50k"], ctx_for(server, tmp_path / "ext"))[0]
    assert res.status == "done"
    root = tmp_path / "ext" / "fsd50k"
    assert (root / "FSD50K.eval_audio" / "2.wav").exists()
    source = json.loads((root / "SOURCE.json").read_text())
    assert source["version"] == "1.0"
    assert all(f["md5_verified"] for f in source["files"])
    assert json.loads((root / ".done").read_text())["clips"] == {"dev": 1, "eval": 1}


def test_fsd50k_split_zip_without_7zip_stops_with_install_hint(server: FakeServer,
                                                               tmp_path: Path) -> None:
    files = fsd_archives()
    files["FSD50K.dev_audio.z01"] = b"part one"
    zenodo(server, files)
    with pytest.raises(dl.SevenZipMissingError, match="winget install 7zip.7zip"):
        dl.download(["fsd50k"], ctx_for(server, tmp_path / "ext"))
    archives = tmp_path / "ext" / "fsd50k" / "archives"
    assert (archives / "FSD50K.dev_audio.z01").exists()  # downloads are kept
    assert not (tmp_path / "ext" / "fsd50k" / ".done").exists()


def test_fsd50k_counts_must_match_the_csvs(server: FakeServer, tmp_path: Path) -> None:
    files = fsd_archives()
    files["FSD50K.eval_audio.zip"] = make_zip({"FSD50K.eval_audio/README": b""})
    zenodo(server, files)
    with pytest.raises(dl.DownloadError, match="eval.csv lists 1 clips"):
        dl.download(["fsd50k"], ctx_for(server, tmp_path / "ext"))


def test_cleanup_lists_only_finished_datasets(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    for ds, done in (("esc50", True), ("fsd50k", False)):
        (ext / ds / "archives").mkdir(parents=True)
        (ext / ds / "archives" / "a.zip").write_bytes(b"x" * 100)
        if done:
            (ext / ds / ".done").write_text("{}")
    dirs = dl.archive_dirs(ext)
    assert [(d.dataset, d.size, d.files) for d in dirs] == [("esc50", 100, 1)]
    assert dl.cleanup_archives(dirs) == 100
    assert not (ext / "esc50" / "archives").exists()
    assert (ext / "fsd50k" / "archives" / "a.zip").exists()
