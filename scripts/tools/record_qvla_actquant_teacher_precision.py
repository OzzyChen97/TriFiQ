#!/usr/bin/env python3
"""Record immutable precision evidence for owned teacher-policy servers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgpack
import zmq


ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "code" / "pi05" / "openpi" / "packages" / "openpi-client" / "src"
if str(CLIENT) not in sys.path:
    sys.path.insert(0, str(CLIENT))


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_environment(pid: int) -> dict[str, str]:
    result = {}
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        if key in (b"GR00T_MODEL_DTYPE", b"OPENPI_MODEL_DTYPE"):
            result[key.decode()] = value.decode(errors="replace")
    return result


def gr00t_metadata(port: int) -> dict[str, Any]:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10_000)
    socket.setsockopt(zmq.SNDTIMEO, 10_000)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.send(msgpack.packb({"endpoint": "get_runtime_info", "data": {}}))
        return msgpack.unpackb(socket.recv(), raw=False)
    finally:
        socket.close(linger=0)
        context.term()


def pi05_metadata(port: int) -> dict[str, Any]:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy("127.0.0.1", port).get_server_metadata()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pid-dir",
        type=Path,
        default=ROOT / "runs" / "qvla_actquant_table1" / "teacher_servers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "qvla_actquant_table1" / "teacher_precision_attestations.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    existing: dict[str, dict[str, Any]] = {}
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        existing = {
            row["server_metadata_sha256"]: row for row in previous.get("servers", [])
        }
    observed = []
    for pid_file in sorted(args.pid_dir.resolve().glob("*.pid")):
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, ValueError, UnicodeDecodeError):
            continue
        match = re.search(r"(?:^| )--port (\d+)(?: |$)", command)
        if not match:
            raise RuntimeError(f"owned teacher lacks --port: {pid_file} {command}")
        port = int(match.group(1))
        model = "gr00t" if "inference_service.py" in command else "pi05"
        if model == "pi05" and "serve_pi05_quant_policy.py" not in command:
            raise RuntimeError(f"unrecognized teacher command: {command}")
        metadata = gr00t_metadata(port) if model == "gr00t" else pi05_metadata(port)
        metadata_sha = (
            str(metadata["metadata_sha256"])
            if model == "gr00t"
            else canonical_hash(metadata)
        )
        section = metadata if model == "gr00t" else metadata.get("openpi_runtime", {})
        precision = section.get("model_dtype") or {}
        environment = process_environment(pid)
        if precision:
            resolved = precision.get("resolved")
            requested = precision.get("requested")
            evidence = "server_runtime_metadata"
        else:
            requested = environment.get("GR00T_MODEL_DTYPE", "bfloat16")
            resolved = {
                "bf16": "bfloat16",
                "bfloat16": "bfloat16",
                "fp16": "float16",
                "float16": "float16",
            }.get(requested.strip().lower())
            evidence = "legacy_gr00t_process_environment_plus_loaded_source_default"
        if resolved not in ("bfloat16", "float16"):
            raise RuntimeError(f"unrecognized teacher precision at port {port}: {precision}")
        linear_dtypes = (
            precision.get("linear_conv_layers_by_weight_dtype")
            if model == "gr00t"
            else precision.get("linear_layers_by_weight_dtype")
        ) or {}
        strict_fp16 = resolved == "float16" and set(linear_dtypes) == {"float16"}
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        record = {
            "server_metadata_sha256": metadata_sha,
            "model": model,
            "requested": requested,
            "resolved": resolved,
            "strict_all_linear_conv_fp16": strict_fp16,
            "evidence": evidence,
            "pid_at_observation": pid,
            "process_start_ticks": int(stat_fields[21]),
            "port_at_observation": port,
            "pid_file": str(pid_file.resolve()),
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        existing[metadata_sha] = record
        observed.append(record)
    if not observed:
        raise RuntimeError("no live owned teacher servers were attested")
    result = {
        "schema_version": 1,
        "kind": "qvla_actquant_teacher_precision_v1",
        "servers": sorted(existing.values(), key=lambda row: row["server_metadata_sha256"]),
    }
    result["inventory_sha256"] = canonical_hash(result["servers"])
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "observed": observed}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
