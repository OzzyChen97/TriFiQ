#!/usr/bin/env python3
"""Download and attest the official Omega-QVLA LIBERO W4A4 packs.

The two Hugging Face revisions are pinned to the commits advertised by the
official release at the time this reproducer was added.  Large tensors remain
outside Git; ``download_manifest.json`` is the portable evidence record.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "checkpoints/omega_qvla"
OFFICIAL_CODE = REPO_ROOT / "external/Omega-QVLA"
SUITES = ("goal", "spatial", "object", "long")
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
RELEASES = {
    "gr00t": {
        "repo_id": "ucmp137538/Omega-QVLA-GR00T-N1.5-LIBERO-W4A4",
        "revision": "01252878485fa9295b53ab833eb13d564f6911ad",
        "pattern": "gr00t_{suite}/quantized.pt",
    },
    "pi05": {
        "repo_id": "ucmp137538/Omega-QVLA-pi05-LIBERO-W4A4",
        "revision": "f3b20e335ca8a03129ab8fe251385c042a13cfe3",
        "pattern": "pi05_{suite}/quantized.pt",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocated_bytes(path: Path) -> int:
    """Return physical bytes without overstating sparse aria2 part files."""
    stat = path.stat()
    return min(stat.st_size, stat.st_blocks * 512)


def code_commit() -> str:
    if not (OFFICIAL_CODE / ".git").is_dir():
        raise RuntimeError(f"official source clone missing: {OFFICIAL_CODE}")
    return subprocess.check_output(
        ["git", "-C", str(OFFICIAL_CODE), "rev-parse", "HEAD"], text=True
    ).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fetch_release_metadata(model: str, endpoint: str) -> list[dict[str, Any]]:
    """Resolve every pinned LFS object through the mirror's blobs API."""
    release = RELEASES[model]
    repo_id = quote(release["repo_id"], safe="/")
    revision = release["revision"]
    url = f"{endpoint}/api/models/{repo_id}/revision/{revision}?blobs=true"
    request = Request(url, headers={"User-Agent": "QuantVLA-Omega-QVLA-reproducer/1"})
    with urlopen(request, timeout=60) as response:
        value = json.load(response)
    require(value.get("sha") == revision, f"{model}: mirror revision drift")
    siblings = {row.get("rfilename"): row for row in value.get("siblings", [])}
    files = []
    for suite in SUITES:
        relative = release["pattern"].format(suite=suite)
        row = siblings.get(relative) or {}
        lfs = row.get("lfs") or {}
        size = int(lfs.get("size") or row.get("size") or 0)
        sha256 = str(lfs.get("sha256") or "")
        require(size > 0, f"{model}/{suite}: missing LFS size from mirror metadata")
        require(re.fullmatch(r"[0-9a-f]{64}", sha256) is not None, f"{model}/{suite}: invalid LFS SHA256")
        files.append(
            {
                "suite": suite,
                "relative": relative,
                "bytes": size,
                "sha256": sha256,
                "blob_id": row.get("blobId"),
            }
        )
    return files


def seed_aria2_partial(model_root: Path, row: dict[str, Any]) -> int:
    """Move a matching huggingface_hub temporary prefix into aria2's part file."""
    relative = Path(row["relative"])
    final = model_root / relative
    part = final.with_name(final.name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.is_file():
        return 0
    if not part.exists():
        cache_dir = model_root / ".cache/huggingface/download" / relative.parent
        candidates = sorted(
            cache_dir.glob(f"*.{row['sha256']}.incomplete"),
            key=lambda path: path.stat().st_size,
            reverse=True,
        ) if cache_dir.is_dir() else []
        if candidates:
            candidates[0].replace(part)
    if not part.exists():
        return 0
    apparent_size = part.stat().st_size
    require(apparent_size <= int(row["bytes"]), f"oversized partial file: {part}")
    return allocated_bytes(part)


def aria2_version() -> str:
    output = subprocess.check_output(["aria2c", "--version"], text=True)
    return output.splitlines()[0].strip()


def aria2_input(
    model: str, model_root: Path, rows: list[dict[str, Any]], endpoint: str
) -> tuple[str, int]:
    release = RELEASES[model]
    blocks = []
    resumed_allocated = 0
    for row in rows:
        relative = Path(row["relative"])
        final = model_root / relative
        if final.is_file():
            require(final.stat().st_size == int(row["bytes"]), f"pack size drift: {final}")
            require(sha256_file(final) == row["sha256"], f"pack SHA drift: {final}")
            continue
        resumed_allocated += seed_aria2_partial(model_root, row)
        encoded_repo = quote(release["repo_id"], safe="/")
        encoded_path = quote(row["relative"], safe="/")
        url = f"{endpoint}/{encoded_repo}/resolve/{release['revision']}/{encoded_path}?download=true"
        blocks.extend(
            [
                url,
                f" dir={final.parent}",
                f" out={final.name}.part",
            ]
        )
    return "\n".join(blocks) + ("\n" if blocks else ""), resumed_allocated


def finalize_aria2_file(model_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    relative = Path(row["relative"])
    final = model_root / relative
    part = final.with_name(final.name + ".part")
    actual = None
    if not final.is_file():
        require(part.is_file(), f"aria2 output missing: {part}")
        require(part.stat().st_size == int(row["bytes"]), f"aria2 size mismatch: {part}")
        actual = sha256_file(part)
        if actual != row["sha256"]:
            quarantine = part.with_name(f"{part.name}.bad-sha256-{actual[:12]}")
            part.replace(quarantine)
            raise RuntimeError(
                f"aria2 SHA mismatch: expected={row['sha256']} actual={actual}; "
                f"quarantined={quarantine}"
            )
        part.replace(final)
    require(final.stat().st_size == int(row["bytes"]), f"pack size drift: {final}")
    if actual is None:
        actual = sha256_file(final)
    require(actual == row["sha256"], f"pack SHA drift: {final}")
    return {
        "suite": row["suite"],
        "path": str(final.relative_to(REPO_ROOT)),
        "bytes": final.stat().st_size,
        "sha256": actual,
        "lfs_blob_id": row.get("blob_id"),
    }


def download_with_aria2(
    models: tuple[str, ...],
    root: Path,
    endpoint: str,
    parallel_files: int,
    connections_per_file: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(shutil.which("aria2c") is not None, "aria2c is not installed")
    metadata = {model: fetch_release_metadata(model, endpoint) for model in models}
    inputs = []
    resumed_allocated_bytes = 0
    for model in models:
        model_root = root / model
        model_root.mkdir(parents=True, exist_ok=True)
        text, resumed = aria2_input(model, model_root, metadata[model], endpoint)
        inputs.append(text)
        resumed_allocated_bytes += resumed
    payload = "".join(inputs)
    if payload:
        command = [
            "aria2c",
            "--input-file=-",
            "--continue=true",
            "--always-resume=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=false",
            "--file-allocation=none",
            f"--max-concurrent-downloads={parallel_files}",
            f"--max-connection-per-server={connections_per_file}",
            f"--split={connections_per_file}",
            "--min-split-size=16M",
            "--connect-timeout=30",
            "--timeout=60",
            "--retry-wait=5",
            "--max-tries=0",
            # The Xet CDN frequently gives each of many parallel ranges less
            # than 64 KiB/s while aggregate throughput remains healthy.  A
            # per-connection floor causes needless reconnect storms.
            "--lowest-speed-limit=0",
            "--disable-ipv6=true",
            "--check-certificate=true",
            "--disk-cache=64M",
            "--summary-interval=30",
            "--console-log-level=notice",
        ]
        subprocess.run(command, input=payload, text=True, check=True)
    records = {}
    for model in models:
        files = [finalize_aria2_file(root / model, row) for row in metadata[model]]
        records[model] = {
            "repo_id": RELEASES[model]["repo_id"],
            "revision": RELEASES[model]["revision"],
            "files": files,
            "total_bytes": sum(row["bytes"] for row in files),
        }
    backend = {
        "name": "aria2",
        "version": aria2_version(),
        "endpoint": endpoint,
        "metadata_api_blobs": True,
        "parallel_files": parallel_files,
        "connections_per_file": connections_per_file,
        "resumed_allocated_bytes": resumed_allocated_bytes,
        "full_lfs_sha256_verified": True,
    }
    return records, backend


def download_one(model: str, root: Path, max_workers: int) -> dict[str, Any]:
    release = RELEASES[model]
    target = root / model
    target.mkdir(parents=True, exist_ok=True)
    allow_patterns = ["README.md"] + [
        release["pattern"].format(suite=suite) for suite in SUITES
    ]
    snapshot_download(
        repo_id=release["repo_id"],
        revision=release["revision"],
        local_dir=target,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )
    files = []
    for suite in SUITES:
        relative = release["pattern"].format(suite=suite)
        path = target / relative
        if not path.is_file():
            raise RuntimeError(f"official pack missing after download: {path}")
        files.append(
            {
                "suite": suite,
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "repo_id": release["repo_id"],
        "revision": release["revision"],
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "gr00t", "pi05"), default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--backend", choices=("aria2", "huggingface"), default="aria2")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--parallel-files", type=int, default=8)
    parser.add_argument("--connections-per-file", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    root = args.root.resolve()
    if REPO_ROOT not in root.parents and root != REPO_ROOT:
        raise ValueError("download root must remain inside the repository workspace")
    selected = tuple(RELEASES) if args.model == "all" else (args.model,)
    if args.backend == "aria2":
        require(args.parallel_files > 0, "parallel-files must be positive")
        require(args.connections_per_file > 0, "connections-per-file must be positive")
        records, backend = download_with_aria2(
            selected,
            root,
            args.endpoint.rstrip("/"),
            args.parallel_files,
            args.connections_per_file,
        )
    else:
        records = {name: download_one(name, root, args.max_workers) for name in selected}
        backend = {
            "name": "huggingface_hub.snapshot_download",
            "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            "max_workers": args.max_workers,
            "full_lfs_sha256_verified": True,
        }
    manifest = {
        "schema_version": 1,
        "kind": "omega_qvla_official_pack_download",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_code": {
            "path": str(OFFICIAL_CODE.relative_to(REPO_ROOT)),
            "commit": code_commit(),
            "remote": "https://github.com/UCMP13753/Omega-QVLA.git",
        },
        "download_backend": backend,
        "models": records,
    }
    manifest_path = root / "download_manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
