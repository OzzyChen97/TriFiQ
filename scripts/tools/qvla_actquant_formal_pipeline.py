#!/usr/bin/env python3
"""Run frozen smoke and formal RoboCasa365 evaluation for QVLA/ActQuant."""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "qvla_actquant_table1"
FORMAL = RUN / "formal"
GROOT_PY = "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
OPENPI_PY = "/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY = "/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
OPENPI_ROOT = ROOT / "code" / "pi05" / "openpi"
OPENPI_CLIENT = OPENPI_ROOT / "packages" / "openpi-client" / "src"
PROTOCOL_PATH = ROOT / "scripts" / "qvla_actquant_table1_protocol.json"
FULL_CONTEXT_PATH = ROOT / "scripts" / "quantvla_full_context_protocol_v2.json"
METHOD_CONFIG = {
    "qvla": "qvla_code_wavg4_a16",
    "actquant": "actquant_4bpw_a16",
}
DEFAULT_ELIGIBLE_GPUS = (3, 4, 5, 6)
SERVER_RESERVE_MIB = 1500
SERVER_ESTIMATE_MIB = {"gr00t": 8500, "pi05": 10500}
WORKER_MIN_FREE_MIB = 2000
MAX_WORKER_ATTEMPTS = 12
WORKER_RETRY_BASE_SECONDS = 15
WORKER_RETRY_MAX_SECONDS = 300
UNITS = {
    "gr00t_atomic_seen": {
        "model": "gr00t",
        "split": "atomic_seen",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000",
    },
    "gr00t_composite_seen": {
        "model": "gr00t",
        "split": "composite_seen",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000",
    },
    "gr00t_composite_unseen": {
        "model": "gr00t",
        "split": "composite_unseen",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000",
    },
    "pi05_all_target": {
        "model": "pi05",
        "split": "all_target",
        "checkpoint": ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def immutable_json(path: Path, value: Any) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"immutable manifest drift: {path}")
        return
    atomic_json(path, value)


def append_event(value: dict[str, Any]) -> None:
    path = FORMAL / "control" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), **value}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_tasks() -> tuple[dict[str, list[str]], dict[str, str]]:
    value = json.loads(FULL_CONTEXT_PATH.read_text(encoding="utf-8"))
    splits = value["table1"]["tasks"]
    task_to_split = {task: split for split, tasks in splits.items() for task in tasks}
    if len(task_to_split) != 50:
        raise ValueError("formal task inventory is not 50 unique tasks")
    return splits, task_to_split


def pack_record(method: str, unit: str) -> dict[str, Any]:
    manifest_path = RUN / "artifacts" / "packs" / method / unit / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Pack manifests use the protocol split name (for example,
    # ``atomic_seen``), while the orchestration directory also prefixes the
    # model family to keep GR00T and pi0.5 units unambiguous.
    expected_model_unit = UNITS[unit]["split"]
    if manifest.get("model_unit") != expected_model_unit or manifest.get("method") != method:
        raise ValueError(f"pack identity mismatch: {manifest_path}")
    if manifest.get("test_results_used") is not False:
        raise ValueError(f"pack used formal feedback: {manifest_path}")
    precision = manifest.get("model_precision") or {}
    expected_dtype = "bfloat16" if method == "qvla" else "float16"
    strict_key = (
        "strict_all_linear_conv_bf16"
        if method == "qvla"
        else "strict_all_linear_conv_fp16"
    )
    if (
        precision.get("resolved") != expected_dtype
        or precision.get(strict_key) is not True
        or manifest.get("activation_compute_dtype") != expected_dtype
    ):
        raise ValueError(f"pack precision attestation mismatch: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if method == "qvla":
        runtime_path = Path(manifest["pack"]["path"])
        runtime_sha = manifest["pack"]["sha256"]
        actual_precision = float(manifest["actual_average_channel_bits"])
        static_bytes = int(manifest["full_component_static_bytes"])
        baseline_bytes = int(manifest["full_component_native_baseline_bytes"])
    else:
        runtime_path = manifest_path
        runtime_sha = manifest_sha
        actual_precision = float(manifest["achieved_bpw"])
        if unit == "pi05_all_target":
            static_bytes = int(manifest["gguf_static_bytes"])
            baseline_bytes = int(manifest["full_fp16_gguf_bytes"])
        else:
            static_bytes = int(manifest["full_component_static_bytes"])
            baseline_bytes = int(manifest["full_component_native_baseline_bytes"])
    if not runtime_path.is_file() or sha256_file(runtime_path) != runtime_sha:
        raise ValueError(f"runtime artifact changed: {runtime_path}")
    if actual_precision > 4.0 + 1e-6:
        raise ValueError(f"precision target exceeded: {manifest_path}")
    return {
        "model_unit": unit,
        "runtime_path": str(runtime_path.resolve()),
        "runtime_sha256": runtime_sha,
        "pack_manifest": str(manifest_path.resolve()),
        "pack_manifest_sha256": manifest_sha,
        "actual_precision": actual_precision,
        "static_bytes": static_bytes,
        "fp16_baseline_bytes": baseline_bytes,
        "compression_ratio": baseline_bytes / static_bytes,
    }


def unit_specs(methods: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    records = []
    selected_methods = methods or tuple(METHOD_CONFIG)
    for method in selected_methods:
        for unit, config in UNITS.items():
            records.append(
                {
                    "id": f"{method}_{unit}",
                    "method": method,
                    "config_id": METHOD_CONFIG[method],
                    "unit": unit,
                    "model": config["model"],
                    "split": config["split"],
                    "checkpoint": str(config["checkpoint"].resolve()),
                    "artifact": pack_record(method, unit),
                }
            )
    return records


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    unknown = sorted(set(methods) - set(METHOD_CONFIG))
    if not methods or unknown:
        raise ValueError(
            f"methods must be a non-empty subset of {sorted(METHOD_CONFIG)}; "
            f"got {value!r}"
        )
    return methods


def parse_gpu_list(value: str) -> tuple[int, ...]:
    try:
        gpus = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as error:
        raise ValueError(f"invalid GPU list: {value!r}") from error
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError(f"GPU list must contain non-negative indices: {value!r}")
    disallowed = sorted(set(gpus) - set(DEFAULT_ELIGIBLE_GPUS))
    if disallowed:
        raise ValueError(
            f"GPUs {disallowed} are outside the hard workspace allowlist "
            f"{list(DEFAULT_ELIGIBLE_GPUS)}"
        )
    return gpus


def gpu_free_mib(eligible_gpus: tuple[int, ...] | None = None) -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = {
        int(row.split(",")[0].strip()): int(row.split(",")[1].strip())
        for row in output.splitlines()
    }
    if eligible_gpus is None:
        eligible_gpus = DEFAULT_ELIGIBLE_GPUS
    missing = sorted(set(eligible_gpus) - set(free))
    if missing:
        raise ValueError(f"eligible GPUs are not visible: {missing}; visible={sorted(free)}")
    return {gpu: free[gpu] for gpu in eligible_gpus}


def make_server_schedule(
    specs: list[dict[str, Any]], replicas: int, eligible_gpus: tuple[int, ...]
) -> dict[str, Any]:
    gpu_label = "-".join(map(str, eligible_gpus))
    path = FORMAL / f"server_schedule.gpus-{gpu_label}.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        expected_artifacts = {
            spec["id"]: spec["artifact"]["runtime_sha256"] for spec in specs
        }
        if (
            any(
                (value.get("artifact_sha256") or {}).get(unit_id) != artifact_sha
                for unit_id, artifact_sha in expected_artifacts.items()
            )
            or value.get("replicas") != replicas
            or value.get("eligible_gpus") != list(eligible_gpus)
        ):
            raise ValueError("frozen server schedule does not match current packs")
        return value
    if replicas < 1:
        raise ValueError("formal protocol requires at least one server replica per model unit")
    if replicas != 1:
        raise ValueError("staged low-VRAM execution currently requires --replicas 1")
    free = gpu_free_mib(eligible_gpus)
    placements = []
    # 24000--24007 are used by the already-running DAPTQ evaluation service
    # in this shared workspace.  Keep this experiment on a disjoint frozen
    # range so runtime attestation can never query another method's server.
    port = 25000
    # Model units are loaded in sequence. The physical GPU is selected from
    # the allowlist immediately before launch, while ports remain frozen.
    for replica in range(replicas):
        for spec in specs:
            placements.append(
                {
                    "instance": f"{spec['id']}_r{replica}",
                    "unit_id": spec["id"],
                    "replica": replica,
                    "gpu": None,
                    "port": port,
                }
            )
            port += 1
    value = {
        "schema_version": 2,
        "kind": "qvla_actquant_formal_server_schedule",
        "immutable": True,
        "execution": "batched_model_units_dynamic_allowed_gpu",
        "replicas": replicas,
        "eligible_gpus": list(eligible_gpus),
        "artifact_sha256": {
            spec["id"]: spec["artifact"]["runtime_sha256"] for spec in specs
        },
        "initial_free_mib": free,
        "reserve_mib": SERVER_RESERVE_MIB,
        "placements": placements,
    }
    immutable_json(path, value)
    return value


def place_server_batch(
    remaining_specs: list[dict[str, Any]],
    schedule: dict[str, Any],
    eligible_gpus: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    while True:
        free = gpu_free_mib(eligible_gpus)
        projected = dict(free)
        selected: list[dict[str, Any]] = []
        placed = []
        occupied_gpus: set[int] = set()
        # Largest-first packing prevents a smaller GR00T service from
        # stranding enough fragmented VRAM to block both pi0.5 services.
        ordered = sorted(
            remaining_specs,
            key=lambda spec: (-SERVER_ESTIMATE_MIB[spec["model"]], spec["id"]),
        )
        for spec in ordered:
            required = SERVER_ESTIMATE_MIB[spec["model"]] + SERVER_RESERVE_MIB
            candidates = [
                gpu
                for gpu, available in projected.items()
                if available >= required and gpu not in occupied_gpus
            ]
            if not candidates:
                continue
            gpu = max(candidates, key=lambda item: (projected[item], -item))
            rows = [row for row in schedule["placements"] if row["unit_id"] == spec["id"]]
            if len(rows) != schedule["replicas"]:
                raise RuntimeError(f"schedule placement count mismatch for {spec['id']}")
            placed.extend({**row, "gpu": gpu} for row in rows)
            selected.append(spec)
            occupied_gpus.add(gpu)
            projected[gpu] -= SERVER_ESTIMATE_MIB[spec["model"]]
        if selected:
            append_event(
                {
                    "event": "server_batch_placed",
                    "unit_ids": [spec["id"] for spec in selected],
                    "placements": placed,
                    "initial_free_mib": free,
                    "projected_free_mib": projected,
                    "eligible_gpus": list(eligible_gpus),
                }
            )
            return selected, {**schedule, "placements": placed}
        atomic_json(
            FORMAL / "control" / "server_wait_status.json",
            {
                "updated_at": now(),
                "remaining_unit_ids": [spec["id"] for spec in remaining_specs],
                "eligible_gpus": list(eligible_gpus),
                "free_mib": free,
                "minimum_required_mib": min(
                    SERVER_ESTIMATE_MIB[spec["model"]] + SERVER_RESERVE_MIB
                    for spec in remaining_specs
                ),
            },
        )
        time.sleep(30)


def clean_quant_env() -> dict[str, str]:
    env = dict(os.environ)
    prefixes = (
        "GR00T_DUQUANT_",
        "OPENPI_DUQUANT_",
        "GR00T_GPTQ",
        "OPENPI_OMEGA_",
        "GR00T_ATM_",
        "OPENPI_ATM_",
        "GR00T_OHB_",
        "OPENPI_OHB_",
        "OPENPI_RUNTIME_SELECTOR_",
        "QVLA_ACTQUANT_",
    )
    for key in list(env):
        if key.startswith(prefixes) or key in {
            "OPENPI_ERRORFOLD_PATH",
            "QUANTVLA_ADAPTER_ONLY",
        }:
            env.pop(key, None)
    # Model services otherwise inherit the host-wide defaults and each
    # PyTorch process creates hundreds of runnable CPU threads.  Four CPU
    # threads per GPU service is sufficient for preprocessing while leaving
    # cores for EGL simulation workers.
    env.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    return env


def server_command(spec: dict[str, Any], placement: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    env = clean_quant_env()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(placement["gpu"]),
            "PYTHONUNBUFFERED": "1",
            "TORCHDYNAMO_DISABLE": "1",
            "QVLA_ACTQUANT_METHOD": spec["method"],
            "QVLA_ACTQUANT_PACK": spec["artifact"]["runtime_path"],
            "QVLA_ACTQUANT_PACK_SHA256": spec["artifact"]["runtime_sha256"],
            "ACTQUANT_ROOT": str(ROOT / "external" / "ActQuant"),
        }
    )
    code_path = f"{ROOT / 'code'}:{ROOT / 'scripts' / 'tools'}"
    env["PYTHONPATH"] = f"{code_path}:{env.get('PYTHONPATH', '')}"
    if spec["model"] == "gr00t":
        env.update(
            {
                "GR00T_CONFIG_ID": spec["config_id"],
                "GR00T_MODEL_DTYPE": (
                    "bfloat16" if spec["method"] == "qvla" else "float16"
                ),
                "GR00T_ATM_ENABLE": "0",
                "GR00T_OHB_ENABLE": "0",
            }
        )
        command = [
            GROOT_PY,
            str(ROOT / "scripts" / "inference_service.py"),
            "--server",
            "--model-path",
            spec["checkpoint"],
            "--data-config",
            "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
            "--embodiment-tag",
            "new_embodiment",
            "--port",
            str(placement["port"]),
            "--denoising-steps",
            "4",
        ]
    else:
        checkpoint_sha = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["models"]["pi05"][
            "checkpoint_sha256"
        ]
        runtime_path = FORMAL / "control" / f"{placement['instance']}.runtime.json"
        env.update(
            {
                "OPENPI_MODEL_DTYPE": (
                    "bfloat16" if spec["method"] == "qvla" else "float16"
                ),
                "OPENPI_CHECKPOINT_SHA256": checkpoint_sha,
                "OPENPI_FORMAL_MODE": "1",
                "OPENPI_FORMAL_FLOW_STEPS": "4",
                "OPENPI_FULL_CONTEXT_PROTOCOL": "1",
                "OPENPI_CONFIG_ID": spec["config_id"],
                "OPENPI_RUNTIME_INFO_PATH": str(runtime_path),
            }
        )
        command = [
            OPENPI_PY,
            "scripts/serve_pi05_quant_policy.py",
            "--env",
            "ROBOCASA",
            "--port",
            str(placement["port"]),
            "--denoising-steps",
            "4",
            "policy:checkpoint",
            "--policy.config",
            "pi05_pretrain_human300",
            "--policy.dir",
            spec["checkpoint"],
        ]
    return command, env


def pid_alive(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(fields) > 2 and fields[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (PermissionError, OSError):
        return True


def process_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def zmq_runtime(port: int, timeout_ms: int = 3000) -> dict[str, Any]:
    import msgpack
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.send(msgpack.packb({"endpoint": "get_runtime_info", "data": {}}))
        return msgpack.unpackb(socket.recv(), raw=False)
    finally:
        socket.close(linger=0)
        context.term()


def http_healthy(port: int) -> bool:
    return subprocess.run(
        ["curl", "-fsS", "--max-time", "2", f"http://127.0.0.1:{port}/healthz"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def validate_runtime(spec: dict[str, Any], runtime: dict[str, Any]) -> None:
    section = runtime.get("openpi_runtime") if spec["model"] == "pi05" else runtime
    if not isinstance(section, dict):
        raise ValueError(f"missing runtime attestation for {spec['id']}")
    reproduction = section.get("qvla_actquant") or {}
    wanted_method = "QVLA-code" if spec["method"] == "qvla" else "ActQuant"
    if reproduction.get("enabled") is not True or reproduction.get("method") != wanted_method:
        raise ValueError(f"reproduction runtime mismatch for {spec['id']}: {reproduction}")
    runtime_sha = reproduction.get(
        "pack_sha256" if spec["method"] == "qvla" else "bundle_manifest_sha256"
    )
    if runtime_sha != spec["artifact"]["runtime_sha256"]:
        raise ValueError(f"runtime pack SHA mismatch for {spec['id']}")
    if reproduction.get("runtime_memory_claim_allowed") is not False:
        raise ValueError(f"runtime memory claim not disabled for {spec['id']}")
    if reproduction.get("latency_claim_allowed") is not False:
        raise ValueError(f"latency claim not disabled for {spec['id']}")
    dtype_inventory = reproduction.get("runtime_dtype_inventory") or {}
    linear_dtypes = dtype_inventory.get("linear_conv_weights_by_dtype") or {}
    if spec["method"] == "qvla":
        if set(linear_dtypes) != {"bfloat16"}:
            raise ValueError(f"QVLA BF16 fixed-precision semantics drift for {spec['id']}: {linear_dtypes}")
    elif set(linear_dtypes) != {"float16"}:
        raise ValueError(f"ActQuant FP16 fixed-precision semantics drift for {spec['id']}: {linear_dtypes}")
    if (section.get("runtime_selector") or {}).get("enabled"):
        raise ValueError(f"selector enabled for {spec['id']}")
    if spec["model"] == "pi05":
        claimed_semantic_sha = section.get("semantic_metadata_sha256")
        stable_runtime = json.loads(json.dumps(runtime, sort_keys=True, default=str))
        stable_section = stable_runtime.get("openpi_runtime") or {}
        stable_section.pop("gpu_memory_bytes", None)
        stable_section.pop("semantic_metadata_sha256", None)
        computed_semantic_sha = canonical_hash(stable_runtime)
        if claimed_semantic_sha != computed_semantic_sha:
            raise ValueError(
                f"pi0.5 semantic metadata SHA mismatch for {spec['id']}: "
                f"{claimed_semantic_sha} != {computed_semantic_sha}"
            )
        expected_dtype = "bfloat16" if spec["method"] == "qvla" else "float16"
        dtype_record = section.get("model_dtype") or {}
        if dtype_record.get("resolved") != expected_dtype:
            raise ValueError(f"pi0.5 model dtype mismatch for {spec['id']}")
        if spec["method"] == "qvla" and dtype_record.get("qvla_uniform_bf16") is not True:
            raise ValueError(f"pi0.5 QVLA uniform-BF16 attestation missing for {spec['id']}")
        forbidden = {
            "duquant": (section.get("duquant") or {}).get("enabled"),
            "omega": (section.get("omega_qvla") or {}).get("enabled"),
            "atm_ohb": (section.get("atm_ohb") or {}).get("enabled"),
            "errorfold": (section.get("errorfold") or {}).get("enabled"),
        }
    else:
        forbidden = {
            "wrappers": int(section.get("wrapped_layers", 0)) != 0,
            "atm": section.get("atm_enabled"),
            "ohb": section.get("ohb_enabled"),
        }
    enabled = [name for name, value in forbidden.items() if value]
    if enabled:
        raise ValueError(f"forbidden mixed methods for {spec['id']}: {enabled}")
    protocol = section.get("protocol") or {}
    frozen_protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_flow = int(frozen_protocol["models"][spec["model"]]["flow_steps"])
    checks = {
        "flow_steps": expected_flow,
        "n_action_steps": 16,
        "replan_steps": 16,
        "split": "target",
        "render": True,
    }
    mismatches = {key: (protocol.get(key), value) for key, value in checks.items() if protocol.get(key) != value}
    if mismatches:
        raise ValueError(f"runtime protocol drift for {spec['id']}: {mismatches}")


def runtime_hash(spec: dict[str, Any], runtime: dict[str, Any]) -> str:
    if spec["model"] == "pi05":
        value = (runtime.get("openpi_runtime") or {}).get("semantic_metadata_sha256")
        if not value:
            raise ValueError("pi0.5 runtime lacks semantic_metadata_sha256")
        return str(value)
    # GR00T publishes a semantic replica-stable identity that deliberately
    # excludes transient CUDA allocator counters.
    value = runtime.get("metadata_sha256")
    if not value:
        raise ValueError("GR00T runtime lacks its stable metadata_sha256")
    return str(value)


def start_servers(specs: list[dict[str, Any]], schedule: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    eligible_gpus = tuple(schedule.get("eligible_gpus", DEFAULT_ELIGIBLE_GPUS))
    disallowed = sorted(set(eligible_gpus) - set(DEFAULT_ELIGIBLE_GPUS))
    invalid_placements = sorted(
        {
            placement.get("gpu")
            for placement in schedule.get("placements", [])
            if placement.get("gpu") not in eligible_gpus
        },
        key=lambda value: (-1 if value is None else int(value)),
    )
    if disallowed or invalid_placements:
        raise ValueError(
            "refusing server schedule outside the hard GPU allowlist: "
            f"eligible={list(eligible_gpus)} disallowed={disallowed} "
            f"invalid_placements={invalid_placements}"
        )
    spec_by_id = {spec["id"]: spec for spec in specs}
    control = FORMAL / "control"
    control.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for replica in range(schedule["replicas"]):
        wave = [row for row in schedule["placements"] if row["replica"] == replica]
        launched = []
        for placement in wave:
            spec = spec_by_id[placement["unit_id"]]
            pid_path = control / f"{placement['instance']}.pid"
            pid = int(pid_path.read_text().strip()) if pid_path.is_file() else -1
            command_line = process_command(pid) if pid > 0 else ""
            wanted_script = (
                "inference_service.py" if spec["model"] == "gr00t" else "serve_pi05_quant_policy.py"
            )
            if pid > 0 and pid_alive(pid) and wanted_script in command_line and f"--port {placement['port']}" in command_line:
                launched.append((placement, spec, pid, True))
                continue
            runtime_path = control / f"{placement['instance']}.runtime.json"
            if runtime_path.exists():
                runtime_path.unlink()
            command, env = server_command(spec, placement)
            log_path = control / f"{placement['instance']}.server.log"
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                command,
                cwd=ROOT if spec["model"] == "gr00t" else OPENPI_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
            launched.append((placement, spec, process.pid, False))
            append_event(
                {
                    "event": "server_started",
                    "instance": placement["instance"],
                    "pid": process.pid,
                    "gpu": placement["gpu"],
                    "port": placement["port"],
                }
            )
            time.sleep(2)
        deadline = time.time() + 1800
        pending = {row[0]["instance"]: row for row in launched}
        startup_errors: dict[str, str] = {}
        while pending and time.time() < deadline:
            for instance, (placement, spec, pid, _reused) in list(pending.items()):
                if not pid_alive(pid):
                    raise RuntimeError(f"formal server exited during startup: {instance}")
                try:
                    if spec["model"] == "gr00t":
                        runtime = zmq_runtime(int(placement["port"]), timeout_ms=5000)
                    else:
                        runtime_path = control / f"{instance}.runtime.json"
                        if not runtime_path.is_file() or not http_healthy(int(placement["port"])):
                            continue
                        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                    validate_runtime(spec, runtime)
                    runtime_path = control / f"{instance}.runtime.json"
                    atomic_json(runtime_path, runtime)
                    metadata_sha = runtime_hash(spec, runtime)
                    record = {
                        **placement,
                        "pid": pid,
                        "runtime_path": str(runtime_path),
                        "runtime_file_sha256": sha256_file(runtime_path),
                        "server_metadata_sha256": metadata_sha,
                    }
                    result[spec["id"]].append(record)
                    del pending[instance]
                    startup_errors.pop(instance, None)
                    append_event({"event": "server_ready", **record})
                except Exception as error:
                    startup_errors[instance] = f"{type(error).__name__}: {error}"
                    continue
            if pending:
                time.sleep(5)
        if pending:
            raise RuntimeError(
                "formal server startup timeout: "
                f"{sorted(pending)}; last validation errors={startup_errors}"
            )
    for spec in specs:
        records = result.get(spec["id"], [])
        if len(records) != schedule["replicas"]:
            raise RuntimeError(f"server replica count mismatch for {spec['id']}")
        hashes = {row["server_metadata_sha256"] for row in records}
        if len(hashes) != 1:
            raise RuntimeError(
                f"replica metadata differs for the same frozen model unit: {spec['id']} {hashes}"
            )
    atomic_json(FORMAL / "server_runtime_inventory.json", dict(result))
    return dict(result)


def stop_servers(schedule: dict[str, Any]) -> None:
    stopped = []
    for placement in schedule.get("placements", []):
        pid_path = FORMAL / "control" / f"{placement['instance']}.pid"
        try:
            pid = int(pid_path.read_text().strip())
        except (OSError, ValueError):
            continue
        if not pid_alive(pid):
            continue
        command = process_command(pid)
        if (
            not ("inference_service.py" in command or "serve_pi05_quant_policy.py" in command)
            or f"--port {placement['port']}" not in command
            or str(ROOT) not in command
        ):
            raise RuntimeError(f"refusing to stop unrelated formal PID {pid}: {command}")
        process_group = os.getpgid(pid)
        if process_group == pid:
            os.killpg(process_group, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        stopped.append(pid)
    deadline = time.time() + 60
    while time.time() < deadline and any(pid_alive(pid) for pid in stopped):
        time.sleep(1)
    remaining = [pid for pid in stopped if pid_alive(pid)]
    if remaining:
        raise RuntimeError(f"formal servers did not exit after SIGTERM: {remaining}")
    append_event({"event": "servers_stopped", "pids": stopped})


def choose_smoke_tasks(spec: dict[str, Any]) -> list[str]:
    splits, _ = load_tasks()
    if spec["model"] == "gr00t":
        return splits[spec["split"]][:5]
    return [
        splits["atomic_seen"][0],
        splits["atomic_seen"][1],
        splits["composite_seen"][0],
        splits["composite_seen"][1],
        splits["composite_unseen"][0],
    ]


def make_jobs(specs: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    _splits, task_to_split = load_tasks()
    jobs = []
    if stage == "smoke1":
        for spec in specs:
            task = choose_smoke_tasks(spec)[0]
            jobs.append(
                {
                    "id": f"smoke1__{spec['id']}",
                    "unit_id": spec["id"],
                    "model": spec["model"],
                    "method": spec["method"],
                    "split": task_to_split[task],
                    "task": task,
                    "seeds": [100],
                }
            )
    elif stage == "smoke50":
        for spec in specs:
            for index, task in enumerate(choose_smoke_tasks(spec)):
                jobs.append(
                    {
                        "id": f"smoke50__{spec['id']}__{index:02d}",
                        "unit_id": spec["id"],
                        "model": spec["model"],
                        "method": spec["method"],
                        "split": task_to_split[task],
                        "task": task,
                        "seeds": list(range(100, 110)),
                    }
                )
    elif stage == "formal":
        for method in ("qvla", "actquant"):
            for task, split in sorted(task_to_split.items(), key=lambda item: (item[1], item[0])):
                unit = {
                    "atomic_seen": "gr00t_atomic_seen",
                    "composite_seen": "gr00t_composite_seen",
                    "composite_unseen": "gr00t_composite_unseen",
                }[split]
                for model in ("gr00t", "pi05"):
                    unit_id = f"{method}_{unit if model == 'gr00t' else 'pi05_all_target'}"
                    for seed_start in range(0, 50, 10):
                        jobs.append(
                            {
                                "id": f"formal__{method}__{model}__{split}__{task}__s{seed_start:02d}",
                                "unit_id": unit_id,
                                "model": model,
                                "method": method,
                                "split": split,
                                "task": task,
                                "seeds": list(range(seed_start, seed_start + 10)),
                            }
                        )
    else:
        raise ValueError(stage)
    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {stage} job IDs")
    if stage == "formal":
        coverage = defaultdict(set)
        for job in jobs:
            arm = (job["method"], job["model"])
            for seed in job["seeds"]:
                key = (job["split"], job["task"], seed)
                if key in coverage[arm]:
                    raise ValueError(f"duplicate formal schedule key: {arm} {key}")
                coverage[arm].add(key)
        if set(coverage) != {
            ("qvla", "gr00t"),
            ("qvla", "pi05"),
            ("actquant", "gr00t"),
            ("actquant", "pi05"),
        } or any(len(keys) != 2500 for keys in coverage.values()):
            raise ValueError("formal schedule is not four disjoint 2,500-episode rows")
    return jobs


def worker_command(
    job: dict[str, Any], server: dict[str, Any], output: Path, egl_gpu: int
) -> tuple[list[str], dict[str, str]]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # RoboCasa workers are process-parallel.  Prevent every worker from also
    # creating a full-machine BLAS/OpenMP pool.
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    seeds = job["seeds"]
    if job["model"] == "gr00t":
        command = [
            ROBOCASA_PY,
            str(ROOT / "scripts" / "run_robocasa365_gr00t_eval.py"),
            "--port",
            str(server["port"]),
            "--task-set",
            job["split"],
            "--tasks",
            job["task"],
            "--n-trials",
            str(len(seeds)),
            "--exact-seeds",
            ",".join(map(str, seeds)),
            "--fresh-env-per-trial",
            "--n-action-steps",
            "16",
            "--paired-action-noise",
            "--out",
            str(output),
        ]
    else:
        env["PYTHONPATH"] = f"{OPENPI_CLIENT}:{env.get('PYTHONPATH', '')}"
        command = [
            ROBOCASA_PY,
            str(ROOT / "scripts" / "run_robocasa365_pi05_eval.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(server["port"]),
            "--config-id",
            METHOD_CONFIG[job["method"]],
            "--task-set",
            job["split"],
            "--tasks",
            job["task"],
            "--trial-seeds",
            ",".join(map(str, seeds)),
            "--split",
            "target",
            "--replan-steps",
            "16",
            "--flow-steps",
            "4",
            "--action-noise-mode",
            "paired",
            "--egl-device",
            str(egl_gpu),
            "--expected-server-metadata-sha256",
            server["server_metadata_sha256"],
            "--out",
            str(output),
        ]
    env["MUJOCO_GL"] = "egl"
    env["MUJOCO_EGL_DEVICE_ID"] = str(egl_gpu)
    return command, env


def validate_worker_output(
    path: Path, job: dict[str, Any], expected_metadata_sha: str
) -> dict[str, Any]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "complete":
            raise ValueError(f"{path}:{line_number}: incomplete row")
        if row.get("task") != job["task"] or int(row.get("seed", -1)) not in job["seeds"]:
            raise ValueError(f"{path}:{line_number}: job-key mismatch")
        if row.get("server_metadata_sha256") != expected_metadata_sha:
            raise ValueError(f"{path}:{line_number}: server metadata mismatch")
        if row.get("runtime_selector_enabled") is not False:
            raise ValueError(f"{path}:{line_number}: selector was enabled")
        if int(row.get("flow_steps", -1)) != 4:
            raise ValueError(f"{path}:{line_number}: flow-step mismatch")
        rows.append(row)
    keys = {(row["task"], int(row["seed"])) for row in rows}
    expected = {(job["task"], seed) for seed in job["seeds"]}
    if keys != expected or len(rows) != len(expected):
        raise ValueError(
            f"{path}: worker coverage mismatch rows={len(rows)} expected={len(expected)}"
        )
    return {
        "rows": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return 0.0


def persist_running_workers(stage: str, running: dict[int, dict[str, Any]]) -> None:
    """Persist enough identity to let a detached supervisor reap only our workers."""
    atomic_json(
        FORMAL / "control" / f"{stage}_running_workers.json",
        {
            "updated_at": now(),
            "stage": stage,
            "workers": [
                {
                    "pid": pid,
                    "process_group": pid,
                    "job_id": item["job"]["id"],
                    "attempt": item["attempt"],
                    "output": str(item["output"].resolve()),
                    "log": item["log"],
                    "egl_gpu": item["egl_gpu"],
                }
                for pid, item in sorted(running.items())
                if item["process"].poll() is None
            ],
        },
    )


def terminate_running_workers(
    stage: str, running: dict[int, dict[str, Any]], *, reason: str
) -> None:
    """Terminate only worker process groups launched by this scheduler instance."""
    live: list[tuple[int, dict[str, Any]]] = []
    for pid, item in running.items():
        if item["process"].poll() is None:
            try:
                process_group = os.getpgid(pid)
                if process_group == pid:
                    os.killpg(process_group, signal.SIGTERM)
                else:
                    item["process"].terminate()
                live.append((pid, item))
            except ProcessLookupError:
                pass
    deadline = time.time() + 30
    while live and time.time() < deadline:
        live = [(pid, item) for pid, item in live if item["process"].poll() is None]
        if live:
            time.sleep(0.5)
    force_killed = []
    for pid, item in live:
        try:
            process_group = os.getpgid(pid)
            if process_group == pid:
                os.killpg(process_group, signal.SIGKILL)
            else:
                item["process"].kill()
            force_killed.append(pid)
        except ProcessLookupError:
            pass
    for item in running.values():
        try:
            item["handle"].close()
        except Exception:
            pass
    if running:
        append_event(
            {
                "event": "worker_pool_cleanup",
                "stage": stage,
                "reason": reason,
                "worker_pids": sorted(running),
                "force_killed_pids": sorted(force_killed),
            }
        )
    running.clear()
    persist_running_workers(stage, running)


def drive_worker_pool(
    *,
    stage: str,
    pending: list[dict[str, Any]],
    running: dict[int, dict[str, Any]],
    gpu_slots: dict[int, int],
    last_launch: dict[int, float],
    receipts: Path,
    logs: Path,
    max_workers: int,
    workers_per_gpu: int,
    eligible_gpus: tuple[int, ...],
    active_unit_ids: set[str],
) -> None:
    while pending or running:
        for pid, item in list(running.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            item["handle"].close()
            gpu_slots[item["egl_gpu"]] -= 1
            job = item["job"]
            failure: Exception | None = None
            checked: dict[str, Any] | None = None
            if return_code != 0:
                failure = RuntimeError(f"exit={return_code}")
            else:
                try:
                    checked = validate_worker_output(
                        item["output"], job, item["server"]["server_metadata_sha256"]
                    )
                except Exception as error:
                    failure = error
            del running[pid]
            if failure is None:
                assert checked is not None
                receipt = {
                    "status": "complete",
                    "job_id": job["id"],
                    "job_sha256": canonical_hash(job),
                    "server_instance": item["server"]["instance"],
                    "server_metadata_sha256": item["server"]["server_metadata_sha256"],
                    "egl_gpu": item["egl_gpu"],
                    "attempt": item["attempt"],
                    "output": str(item["output"]),
                    "output_sha256": checked["sha256"],
                    "rows": checked["rows"],
                    "successes": checked["successes"],
                }
                atomic_json(receipts / f"{job['id']}.json", receipt)
                append_event({"event": "worker_complete", **receipt})
                continue
            if item["attempt"] >= MAX_WORKER_ATTEMPTS:
                append_event(
                    {
                        "event": "worker_attempts_exhausted",
                        "stage": stage,
                        "job_id": job["id"],
                        "attempt": item["attempt"],
                        "error": repr(failure),
                        "log": item["log"],
                    }
                )
                raise RuntimeError(
                    f"worker failed after {MAX_WORKER_ATTEMPTS} attempts: "
                    f"{job['id']}; log={item['log']}"
                ) from failure
            retry_delay = min(
                WORKER_RETRY_MAX_SECONDS,
                WORKER_RETRY_BASE_SECONDS * (2 ** max(0, item["attempt"] - 1)),
            )
            pending.append(
                {
                    "job": job,
                    "server": item["server"],
                    "output": item["output"],
                    "attempt": item["attempt"],
                    "retry_not_before": time.monotonic() + retry_delay,
                }
            )
            append_event(
                {
                    "event": "worker_retry",
                    "stage": stage,
                    "job_id": job["id"],
                    "attempt": item["attempt"],
                    "retry_delay_seconds": retry_delay,
                    "error": repr(failure),
                }
            )

        if len(running) < max_workers and pending and mem_available_gib() >= 48:
            free = gpu_free_mib(eligible_gpus)
            launched_gpus = set()
            for item in list(pending):
                if len(running) >= max_workers:
                    break
                if time.monotonic() < item.get("retry_not_before", 0.0):
                    continue
                candidates = [
                    gpu
                    for gpu, available in free.items()
                    if available >= WORKER_MIN_FREE_MIB
                    and gpu_slots[gpu] < workers_per_gpu
                    and gpu not in launched_gpus
                    and time.monotonic() - last_launch[gpu] >= 10
                ]
                if not candidates:
                    break
                gpu = max(candidates, key=lambda value: (free[value], -gpu_slots[value], -value))
                item["attempt"] += 1
                job = item["job"]
                output = item["output"]
                output.parent.mkdir(parents=True, exist_ok=True)
                command, env = worker_command(job, item["server"], output, gpu)
                log = logs / f"{job['id']}.attempt{item['attempt']}.log"
                handle = log.open("a", encoding="utf-8", buffering=1)
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except Exception:
                    handle.close()
                    raise
                running[process.pid] = {
                    **item,
                    "process": process,
                    "handle": handle,
                    "egl_gpu": gpu,
                    "log": str(log),
                }
                pending.remove(item)
                gpu_slots[gpu] += 1
                launched_gpus.add(gpu)
                last_launch[gpu] = time.monotonic()
                append_event(
                    {
                        "event": "worker_started",
                        "stage": stage,
                        "job_id": job["id"],
                        "pid": process.pid,
                        "egl_gpu": gpu,
                        "server": item["server"]["instance"],
                        "attempt": item["attempt"],
                    }
                )
        persist_running_workers(stage, running)
        monotonic_now = time.monotonic()
        retry_waits = [
            max(0.0, item.get("retry_not_before", 0.0) - monotonic_now)
            for item in pending
            if item.get("retry_not_before", 0.0) > monotonic_now
        ]
        atomic_json(
            FORMAL / "control" / f"{stage}_status.json",
            {
                "updated_at": now(),
                "stage": stage,
                "pending": len(pending),
                "running": len(running),
                "completed": len(list(receipts.glob("*.json"))),
                "cooling_down": len(retry_waits),
                "next_retry_seconds": min(retry_waits) if retry_waits else 0.0,
                "gpu_worker_slots": dict(gpu_slots),
                "eligible_gpus": list(eligible_gpus),
                "active_unit_ids": sorted(active_unit_ids),
                "mem_available_gib": mem_available_gib(),
            },
        )
        if pending or running:
            time.sleep(10)


def run_jobs(
    specs: list[dict[str, Any]],
    servers: dict[str, list[dict[str, Any]]],
    stage: str,
    *,
    max_workers: int,
    workers_per_gpu: int,
    eligible_gpus: tuple[int, ...],
    active_unit_ids: set[str] | None = None,
) -> None:
    manifest_path = FORMAL / f"{stage}_jobs.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        jobs = manifest.get("jobs") or []
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != "qvla_actquant_rollout_jobs"
            or manifest.get("immutable") is not True
            or manifest.get("stage") != stage
            or manifest.get("formal_feedback_allowed") is not False
            or manifest.get("jobs_sha256") != canonical_hash(jobs)
        ):
            raise ValueError(f"invalid immutable job manifest: {manifest_path}")
    else:
        jobs = make_jobs(specs, stage)
        manifest = {
            "schema_version": 1,
            "kind": "qvla_actquant_rollout_jobs",
            "immutable": True,
            "stage": stage,
            "jobs": jobs,
            "jobs_sha256": canonical_hash(jobs),
            "formal_feedback_allowed": False,
        }
        immutable_json(manifest_path, manifest)
    output_root = FORMAL / ("raw" if stage == "formal" else "smoke") / stage
    receipts = FORMAL / "control" / "receipts" / stage
    logs = FORMAL / "control" / "workers" / stage
    receipts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    pending = []
    for job in jobs:
        output = output_root / job["method"] / job["model"] / f"{job['id']}.jsonl"
        receipt = receipts / f"{job['id']}.json"
        if receipt.is_file():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved.get("job_sha256") != canonical_hash(job):
                raise ValueError(f"receipt job drift: {receipt}")
            checked = validate_worker_output(output, job, saved["server_metadata_sha256"])
            if checked["sha256"] != saved["output_sha256"]:
                raise ValueError(f"completed output changed: {output}")
            continue
        if active_unit_ids is not None and job["unit_id"] not in active_unit_ids:
            continue
        replicas = sorted(servers[job["unit_id"]], key=lambda row: row["replica"])
        replica_index = int(hashlib.sha256(job["id"].encode()).hexdigest(), 16) % len(replicas)
        server = replicas[replica_index]
        pending.append({"job": job, "server": server, "output": output, "attempt": 0})
    # Preserve the immutable job manifest but launch round-robin across model
    # units.  A grouped launch order leaves later servers idle for a large
    # fraction of smoke/formal evaluation and is especially costly when each
    # unit has one memory-constrained replica.
    unit_order = list(dict.fromkeys(item["job"]["unit_id"] for item in pending))
    by_unit = {
        unit_id: [item for item in pending if item["job"]["unit_id"] == unit_id]
        for unit_id in unit_order
    }
    pending = [
        by_unit[unit_id][index]
        for index in range(max((len(rows) for rows in by_unit.values()), default=0))
        for unit_id in unit_order
        if index < len(by_unit[unit_id])
    ]
    running: dict[int, dict[str, Any]] = {}
    gpu_slots: dict[int, int] = defaultdict(int)
    last_launch: dict[int, float] = defaultdict(lambda: 0.0)
    selected_unit_ids = active_unit_ids or {spec["id"] for spec in specs}
    try:
        drive_worker_pool(
            stage=stage,
            pending=pending,
            running=running,
            gpu_slots=gpu_slots,
            last_launch=last_launch,
            receipts=receipts,
            logs=logs,
            max_workers=max_workers,
            workers_per_gpu=workers_per_gpu,
            eligible_gpus=eligible_gpus,
            active_unit_ids=selected_unit_ids,
        )
    finally:
        terminate_running_workers(stage, running, reason="scheduler_exit")
    completed_receipts = len(list(receipts.glob("*.json")))
    append_event(
        {
            "event": (
                "rollout_stage_complete"
                if completed_receipts == len(jobs)
                else "rollout_unit_stage_complete"
            ),
            "stage": stage,
            "jobs": len(jobs),
            "completed_receipts": completed_receipts,
            "active_unit_ids": sorted(selected_unit_ids),
        }
    )


def artifact_for_arm(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_unit": record["model_unit"],
        "path": record["runtime_path"],
        "sha256": record["runtime_sha256"],
        "pack_manifest": record["pack_manifest"],
        "pack_manifest_sha256": record["pack_manifest_sha256"],
        "actual_precision": record["actual_precision"],
        "static_bytes": record["static_bytes"],
        "fp16_baseline_bytes": record["fp16_baseline_bytes"],
        "compression_ratio": record["compression_ratio"],
    }


def completed_result_server_hashes(method: str, model: str) -> list[str]:
    """Bind a final arm to the servers that produced its completed rows.

    A resumed pipeline may start replacement servers after all rollout receipts
    already exist.  Their runtime metadata can differ from the metadata frozen
    into the completed rows, so using the replacement inventory would make an
    otherwise unchanged immutable arm manifest drift during finalization.
    """
    paths = sorted((FORMAL / "raw" / "formal" / method / model).glob("*.jsonl"))
    hashes: set[str] = set()
    rows = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            metadata_sha = str(record.get("server_metadata_sha256", ""))
            if len(metadata_sha) != 64:
                raise ValueError(f"malformed server metadata SHA in {path}")
            hashes.add(metadata_sha)
            rows += 1
    if rows != 2500:
        raise ValueError(f"{method}/{model} finalization requires 2500 rows, got {rows}")
    if not hashes:
        raise ValueError(f"{method}/{model} has no completed server metadata")
    return sorted(hashes)


def make_candidate_arm_manifests(
    specs: list[dict[str, Any]], _servers: dict[str, list[dict[str, Any]]]
) -> list[Path]:
    protocol_sha = sha256_file(PROTOCOL_PATH)
    outputs = []
    for method in sorted({spec["method"] for spec in specs}):
        for model in ("gr00t", "pi05"):
            selected = [spec for spec in specs if spec["method"] == method and spec["model"] == model]
            artifacts = [artifact_for_arm(spec["artifact"]) for spec in selected]
            # Final evidence is bound to the runtime metadata recorded in the
            # completed episodes, not to replacement servers started by a
            # later no-op resume attempt.
            metadata = completed_result_server_hashes(method, model)
            static_bytes = sum(row["static_bytes"] for row in artifacts)
            baseline_bytes = sum(row["fp16_baseline_bytes"] for row in artifacts)
            path = FORMAL / "arm_manifests" / f"{method}_{model}.json"
            value = {
                "schema_version": 1,
                "kind": "qvla_actquant_formal_arm",
                "immutable": True,
                "method": method,
                "model": model,
                "display_name": (
                    "QVLA-code Wavg4/A16" if method == "qvla" else "ActQuant 4.0 BPW/A16"
                ),
                "protocol": str(PROTOCOL_PATH),
                "protocol_sha256": protocol_sha,
                "source_provenance": str((RUN / "provenance.json").resolve()),
                "source_provenance_sha256": sha256_file(RUN / "provenance.json"),
                "flow_steps": 4,
                "formal_seeds": list(range(50)),
                "episodes": 2500,
                "test_feedback_allowed": False,
                "artifacts": artifacts,
                "allowed_server_metadata_sha256": metadata,
                "result_globs": [
                    str((FORMAL / "raw" / "formal" / method / model / "*.jsonl").resolve())
                ],
                "storage": {
                    "static_bytes_total_deployed_checkpoint_set": static_bytes,
                    "static_gib_total_deployed_checkpoint_set": static_bytes / 1024**3,
                    "fp16_baseline_bytes_total_deployed_checkpoint_set": baseline_bytes,
                    "compression_ratio_total_deployed_checkpoint_set": baseline_bytes / static_bytes,
                    "checkpoint_count": len(artifacts),
                    "scope": (
                        "sum across three split-specific checkpoints"
                        if model == "gr00t"
                        else "single global checkpoint"
                    ),
                },
            }
            immutable_json(path, value)
            outputs.append(path)
    return outputs


def raw_server_hashes(glob_expression: str) -> list[str]:
    hashes = set()
    for path in ROOT.glob(glob_expression):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                hashes.add(str(json.loads(line)["server_metadata_sha256"]))
    if not hashes:
        raise ValueError(f"baseline glob matched no metadata: {glob_expression}")
    return sorted(hashes)


def make_baseline_manifests() -> dict[str, Path]:
    protocol_sha = sha256_file(PROTOCOL_PATH)
    configs = {
        "gr00t": "runs/full_context_v2/table1/results/full_context_v2/**/*.jsonl",
        "pi05": "runs/full_context_v2/pi05_table1/results/full_context_w4a8_dynamic_profile/*.jsonl",
    }
    outputs = {}
    for model, expression in configs.items():
        path = FORMAL / "arm_manifests" / f"dypac_{model}.json"
        value = {
            "schema_version": 1,
            "kind": "qvla_actquant_formal_arm",
            "immutable": True,
            "method": "dypac",
            "model": model,
            "display_name": "DyPAC-VLA",
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": protocol_sha,
            "flow_steps": 4,
            "test_feedback_allowed": False,
            "allowed_server_metadata_sha256": raw_server_hashes(expression),
            "result_globs": [str((ROOT / expression).resolve())],
            "comparison_role": "compatible paired baseline",
        }
        immutable_json(path, value)
        outputs[model] = path
    return outputs


def run_checked(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", buffering=1) as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def canonicalize_and_aggregate(candidate_manifests: list[Path], baseline_manifests: dict[str, Path]) -> None:
    tool = ROOT / "scripts" / "tools" / "qvla_actquant_formal_results.py"
    canonical_dir = FORMAL / "canonical"
    candidates = []
    for manifest in [*candidate_manifests, *baseline_manifests.values()]:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        output = canonical_dir / f"{value['method']}_{value['model']}.jsonl"
        run_checked(
            [GROOT_PY, str(tool), "canonicalize", "--arm-manifest", str(manifest), "--out", str(output)],
            FORMAL / "control" / "canonicalize.log",
        )
        if value["method"] != "dypac":
            candidates.append(output)
    command = [GROOT_PY, str(tool), "aggregate"]
    for path in sorted(candidates):
        command += ["--candidate", str(path)]
    for model in ("gr00t", "pi05"):
        command += ["--baseline", f"{model}={canonical_dir / f'dypac_{model}.jsonl'}"]
    command += ["--bootstrap", "10000", "--out", str(FORMAL / "aggregate.json")]
    run_checked(command, FORMAL / "control" / "aggregate.log")


def execute(
    max_workers: int,
    workers_per_gpu: int,
    replicas: int,
    eligible_gpus: tuple[int, ...],
    methods: tuple[str, ...],
) -> None:
    lock_path = FORMAL / "control" / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("formal pipeline is already running") from error
        specs = unit_specs(methods)
        schedule = make_server_schedule(specs, replicas, eligible_gpus)
        server_inventory: dict[str, list[dict[str, Any]]] = {}
        append_event(
            {
                "event": "gpu_allowlist_frozen",
                "eligible_gpus": list(eligible_gpus),
                "execution": "batched_model_units",
                "methods": list(methods),
            }
        )
        remaining_specs = list(specs)
        while remaining_specs:
            active_specs, active_schedule = place_server_batch(
                remaining_specs, schedule, eligible_gpus
            )
            servers: dict[str, list[dict[str, Any]]] = {}
            try:
                servers = start_servers(active_specs, active_schedule)
                server_inventory.update(servers)
                for stage in ("smoke1", "smoke50", "formal"):
                    run_jobs(
                        specs,
                        servers,
                        stage,
                        max_workers=min(max_workers, 8) if stage == "smoke1" else max_workers,
                        workers_per_gpu=workers_per_gpu,
                        eligible_gpus=eligible_gpus,
                        active_unit_ids={spec["id"] for spec in active_specs},
                    )
            finally:
                if servers:
                    stop_servers(active_schedule)
            active_ids = {spec["id"] for spec in active_specs}
            remaining_specs = [spec for spec in remaining_specs if spec["id"] not in active_ids]
        atomic_json(FORMAL / "server_runtime_inventory.json", server_inventory)
        candidates = make_candidate_arm_manifests(specs, server_inventory)
        baselines = make_baseline_manifests()
        canonicalize_and_aggregate(candidates, baselines)
        run_checked(
            [
                GROOT_PY,
                str(
                    ROOT
                    / "scripts"
                    / "tools"
                    / "render_qvla_actquant_paper_extension.py"
                ),
            ],
            FORMAL / "control" / "paper_extension.log",
        )
        run_checked(
            ["make", "-C", str(ROOT / "docs" / "gdsq_vla_iclr2027"), "check"],
            FORMAL / "control" / "paper_check.log",
        )
        atomic_json(
            FORMAL / "complete.json",
            {
                "completed_at": now(),
                "status": "complete",
                "methods": list(methods),
                "new_formal_episodes": 2500
                * len({(spec["method"], spec["model"]) for spec in specs}),
                "aggregate": str(FORMAL / "aggregate.json"),
                "aggregate_sha256": sha256_file(FORMAL / "aggregate.json"),
            },
        )


def status() -> dict[str, Any]:
    result = {}
    for stage in ("smoke1", "smoke50", "formal"):
        path = FORMAL / "control" / f"{stage}_status.json"
        result[stage] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    complete = FORMAL / "complete.json"
    result["complete"] = json.loads(complete.read_text(encoding="utf-8")) if complete.is_file() else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "stop-servers"))
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_CONFIG),
        help="comma-separated method subset (qvla,actquant)",
    )
    parser.add_argument(
        "--eligible-gpus",
        default=",".join(map(str, DEFAULT_ELIGIBLE_GPUS)),
        help="comma-separated physical GPU allowlist (default: 3,4,5,6)",
    )
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return
    if args.command == "stop-servers":
        for path in sorted(FORMAL.glob("server_schedule*.json")):
            stop_servers(json.loads(path.read_text(encoding="utf-8")))
        return
    if args.max_workers < 1 or args.workers_per_gpu < 1:
        raise ValueError("worker concurrency must be positive")
    eligible_gpus = parse_gpu_list(args.eligible_gpus)
    methods = parse_methods(args.methods)
    execute(args.max_workers, args.workers_per_gpu, args.replicas, eligible_gpus, methods)


if __name__ == "__main__":
    main()
