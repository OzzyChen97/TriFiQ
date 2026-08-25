from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "scripts/tools/download_omega_qvla_packs.py"
    spec = importlib.util.spec_from_file_location("download_omega_qvla_packs", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DOWNLOAD = load_module()


class FakeResponse:
    def __init__(self, value: dict):
        self.payload = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *_args):
        return self.payload


def release_payload(model: str) -> dict:
    release = DOWNLOAD.RELEASES[model]
    siblings = []
    for index, suite in enumerate(DOWNLOAD.SUITES, start=1):
        content = f"{model}-{suite}".encode()
        siblings.append(
            {
                "rfilename": release["pattern"].format(suite=suite),
                "blobId": f"blob-{index}",
                "lfs": {
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
            }
        )
    return {"sha": release["revision"], "siblings": siblings}


def test_mirror_metadata_is_revision_and_lfs_attested(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return FakeResponse(release_payload("gr00t"))

    monkeypatch.setattr(DOWNLOAD, "urlopen", fake_urlopen)
    rows = DOWNLOAD.fetch_release_metadata("gr00t", "https://hf-mirror.example")
    assert [row["suite"] for row in rows] == list(DOWNLOAD.SUITES)
    assert all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in rows)
    assert DOWNLOAD.RELEASES["gr00t"]["revision"] in seen["url"]
    assert seen["url"].endswith("?blobs=true")
    assert seen["timeout"] == 60


def test_mirror_revision_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    value = release_payload("pi05")
    value["sha"] = "0" * 40
    monkeypatch.setattr(DOWNLOAD, "urlopen", lambda *_args, **_kwargs: FakeResponse(value))
    with pytest.raises(RuntimeError, match="mirror revision drift"):
        DOWNLOAD.fetch_release_metadata("pi05", "https://hf-mirror.example")


def test_huggingface_partial_is_reused_and_url_remains_pinned(tmp_path: Path) -> None:
    model = "gr00t"
    release = DOWNLOAD.RELEASES[model]
    content = b"partial-prefix"
    full = b"complete-pack-with-prefix-room"
    row = {
        "suite": "goal",
        "relative": release["pattern"].format(suite="goal"),
        "bytes": len(full),
        "sha256": hashlib.sha256(full).hexdigest(),
        "blob_id": "blob",
    }
    cache = tmp_path / ".cache/huggingface/download/gr00t_goal"
    cache.mkdir(parents=True)
    source = cache / f"prefix.{row['sha256']}.incomplete"
    source.write_bytes(content)

    payload, resumed = DOWNLOAD.aria2_input(model, tmp_path, [row], "https://hf-mirror.example")
    part = tmp_path / "gr00t_goal/quantized.pt.part"
    assert resumed == len(content)
    assert not source.exists()
    assert part.read_bytes() == content
    assert release["revision"] in payload
    assert "https://hf-mirror.example/" in payload
    assert " out=quantized.pt.part" in payload


def test_final_promotion_requires_exact_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DOWNLOAD, "REPO_ROOT", tmp_path)
    original_sha256_file = DOWNLOAD.sha256_file
    sha_calls = []

    def counted_sha256(path: Path) -> str:
        sha_calls.append(path)
        return original_sha256_file(path)

    monkeypatch.setattr(DOWNLOAD, "sha256_file", counted_sha256)
    content = b"verified-pack"
    row = {
        "suite": "goal",
        "relative": "gr00t_goal/quantized.pt",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "blob_id": "blob",
    }
    part = tmp_path / "gr00t_goal/quantized.pt.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(content)
    record = DOWNLOAD.finalize_aria2_file(tmp_path, row)
    assert not part.exists()
    assert (tmp_path / row["relative"]).read_bytes() == content
    assert record["sha256"] == row["sha256"]
    assert len(sha_calls) == 1


def test_sparse_partial_resume_counts_allocated_not_apparent_bytes(tmp_path: Path) -> None:
    row = {
        "suite": "goal",
        "relative": "gr00t_goal/quantized.pt",
        "bytes": 8 * 1024 * 1024,
        "sha256": "a" * 64,
        "blob_id": "blob",
    }
    part = tmp_path / "gr00t_goal/quantized.pt.part"
    part.parent.mkdir(parents=True)
    with part.open("wb") as handle:
        handle.seek(4 * 1024 * 1024)
        handle.write(b"x")
    resumed = DOWNLOAD.seed_aria2_partial(tmp_path, row)
    assert resumed == DOWNLOAD.allocated_bytes(part)
    assert resumed < part.stat().st_size
