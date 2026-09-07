#!/usr/bin/env python3
"""GPU7-only resumable runner for the GR00T DPAC predictive-validity study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import msgpack
import numpy as np
import zmq

from quantvla_predictive_validity import (
    PROTOCOL,
    artifact,
    atomic_json,
    protocol_attestation,
    require_protocol_attestation,
    sha256_file,
)


REPO = Path(__file__).resolve().parents[2]
ROOT = (REPO / PROTOCOL["execution"]["root"]).resolve()
LIBRARY = ROOT / "mask_library.json"
OFFLINE_ROOT = ROOT / "offline"
CLOSED_ROOT = ROOT / "closed_loop"
SMOKE_ROOT = ROOT / "smoke"
REPORT_ROOT = ROOT / "report"
GROOT_PY = Path("/home1/gyy/probe/miniforge3/envs/groot_test/bin/python")
ROBOCASA_PY = Path("/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python")
SERVER_SCRIPT = REPO / "scripts/run_robocasa365_quant_serve.sh"
CLIENT_SCRIPT = REPO / "scripts/run_robocasa365_gr00t_eval.py"
PORT = 27007
GPU = 7
SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")


def load_library() -> dict[str, Any]:
    value = json.loads(LIBRARY.read_text(encoding="utf-8"))
    require_protocol_attestation(value, source=str(LIBRARY))
    if value.get("kind") != "gr00t_dpac_predictive_mask_library":
        raise ValueError("wrong predictive mask-library kind")
    if int(value.get("candidate_count", -1)) != 60:
        raise ValueError("predictive mask library must contain exactly 60 masks")
    return value


def gpu_free_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi", f"--id={GPU}", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ], text=True,
    ).strip()
    return int(output.splitlines()[0].strip())


def wait_for_gpu() -> None:
    required = int(PROTOCOL["runtime"]["minimum_free_gpu_mib"])
    while True:
        free = gpu_free_mib()
        if free >= required:
            print(f"[predictive/coverage] GPU7 preflight passed: free={free} MiB", flush=True)
            return
        print(
            f"[predictive/coverage] GPU7 waiting: free={free} MiB required={required} MiB; "
            "no process will be evicted",
            flush=True,
        )
        time.sleep(30)


def clean_server_env() -> dict[str, str]:
    env = dict(os.environ)
    prefixes = (
        "GR00T_DUQUANT_", "GR00T_GPTQ", "GR00T_RTN", "GR00T_ATM_",
        "GR00T_OHB_", "GR00T_RUNTIME_SELECTOR", "GR00T_ERRORFOLD",
        "GR00T_PREDICTIVE_", "OMEGA_QVLA_", "QVLA_ACTQUANT_", "DAPTQ_",
    )
    for key in list(env):
        if key.startswith(prefixes):
            env.pop(key, None)
    env["PYTHONPATH"] = (
        f"{REPO / 'code'}:{REPO / 'scripts/tools'}:{env.get('PYTHONPATH', '')}"
    )
    return env


def clean_client_env() -> dict[str, str]:
    env = dict(os.environ)
    blocked = {REPO.resolve(), (REPO / "code").resolve()}
    safe = []
    for value in env.get("PYTHONPATH", "").split(os.pathsep):
        if not value:
            continue
        path = Path(value)
        resolved = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
        if resolved not in blocked:
            safe.append(value)
    if safe:
        env["PYTHONPATH"] = os.pathsep.join(safe)
    else:
        env.pop("PYTHONPATH", None)
    env["PYTHONUSERBASE"] = str(REPO / ".pyuserbase")
    env["MUJOCO_EGL_DEVICE_ID"] = str(GPU)
    env.setdefault("NUMBA_CACHE_DIR", str(ROOT / "numba_cache"))
    return env


def endpoint(port: int, name: str, timeout_ms: int = 5000) -> dict[str, Any]:
    def encode(obj):
        if isinstance(obj, np.ndarray):
            out = io.BytesIO()
            np.save(out, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": out.getvalue()}
        return obj

    def decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(f"tcp://localhost:{port}")
    try:
        socket.send(msgpack.packb({"endpoint": name, "data": {}}, default=encode))
        return msgpack.unpackb(socket.recv(), object_hook=decode)
    finally:
        socket.close(linger=0)
        context.term()


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def start_server(library: dict[str, Any], split: str, *, predictive: bool, label: str):
    env = clean_server_env()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "GR00T_GPU": str(GPU),
            "GR00T_PORT": str(PORT),
            "GR00T_DENOISING_STEPS": "4",
            "GR00T_CONFIG_ID": label,
            "GR00T_MODEL_PATH": library["split_sources"][split]["checkpoint"]["path"],
            "GR00T_DATA_CONFIG": (
                "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig"
            ),
            "GR00T_ATM_ENABLE": "0",
            "GR00T_OHB_ENABLE": "0",
        }
    )
    if predictive:
        env.update(
            {
                "GR00T_DUQUANT_PLAN": library["full_w4_plan"]["path"],
                "GR00T_DUQUANT_PACKDIR": library["split_sources"][split]["identity_pack"]["path"],
                "GR00T_DUQUANT_HESSIAN_W4_PATH": library["split_sources"][split]["hessian_w4"]["path"],
                "GR00T_DUQUANT_ABITS": "8",
                "GR00T_DUQUANT_ACT_DYNAMIC": "1",
                "GR00T_DUQUANT_FUSED": "1",
                "QUANTVLA_ADAPTER_ONLY": "1",
                "GR00T_PREDICTIVE_MASK_MANIFEST": str(LIBRARY),
            }
        )
        command = ["bash", str(SERVER_SCRIPT)]
    else:
        env.pop("QUANTVLA_ADAPTER_ONLY", None)
        command = [
            str(GROOT_PY), str(REPO / "scripts/inference_service.py"), "--server",
            "--model-path", env["GR00T_MODEL_PATH"],
            "--data-config", env["GR00T_DATA_CONFIG"],
            "--embodiment-tag", "new_embodiment", "--port", str(PORT),
            "--denoising-steps", "4",
        ]
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    handle = (logs / f"server_{label}.log").open("a", buffering=1)
    process = subprocess.Popen(
        command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 1800
    last_error: Exception | None = None
    try:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server {label} exited with {process.returncode}")
            try:
                info = endpoint(PORT, "get_runtime_info", timeout_ms=120_000)
                if "error" in info:
                    raise RuntimeError(str(info["error"]))
                if predictive:
                    runtime = info.get("predictive_mask_runtime") or {}
                    contract = info.get("quantization_contract") or {}
                    if (
                        runtime.get("manifest_sha256") != sha256_file(LIBRARY)
                        or runtime.get("candidate_count") != 60
                        or info.get("wrapped_layers") != 116
                        or contract.get("logical_profile")
                        != "gr00t_predictive_mask_superset_w4a8"
                    ):
                        raise RuntimeError(f"predictive runtime attestation failed: {info}")
                elif (
                    info.get("wrapped_layers") != 0
                    or (info.get("quantization_contract") or {}).get("logical_profile") != "fp16"
                    or (info.get("predictive_mask_runtime") or {}).get("enabled")
                ):
                    raise RuntimeError(f"native FP16 runtime attestation failed: {info}")
                runtime_path = ROOT / "runtime" / f"{label}.json"
                atomic_json(runtime_path, info)
                print(f"[predictive/coverage] server ready: {label}", flush=True)
                return process, handle
            except Exception as exc:
                last_error = exc
                time.sleep(3)
        raise RuntimeError(f"server {label} readiness timeout: {last_error}")
    except BaseException:
        stop_process(process)
        handle.close()
        raise


def rows_in(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verify_client_file(
    path: Path, tasks: list[str], *, mask_id: str | None, library: dict[str, Any]
) -> None:
    rows = rows_in(path)
    expected = {(task, 70) for task in tasks}
    actual = [(str(row.get("task")), int(row.get("seed", -1))) for row in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError(
            f"{path}: incomplete/duplicate coverage {len(actual)}/{len(expected)}"
        )
    plan_hash = None
    if mask_id is not None:
        plan_hash = next(
            row["sha256"] for row in library["candidates"]
            if row["candidate_id"] == mask_id
        )
    for row in rows:
        if row.get("status") != "complete" or not isinstance(row.get("success"), bool):
            raise RuntimeError(f"{path}: invalid committed row")
        if row.get("predictive_mask_id") != mask_id:
            raise RuntimeError(f"{path}: predictive mask ID drift")
        if mask_id is not None and (
            row.get("predictive_plan_sha256") != plan_hash
            or row.get("predictive_library_sha256") != sha256_file(LIBRARY)
        ):
            raise RuntimeError(f"{path}: predictive hash drift")


def client_command(
    tasks: list[str], path: Path, *, mask_id: str | None
) -> list[str]:
    command = [
        str(ROBOCASA_PY), str(CLIENT_SCRIPT), "--port", str(PORT),
        "--server-timeout-ms", "180000", "--task-set", "custom",
        "--tasks", ",".join(tasks), "--n-trials", "1",
        "--shared-exact-seed", "70", "--fresh-env-per-trial",
        "--max-steps", "0", "--n-action-steps", "16",
        "--paired-action-noise", "--out", str(path),
    ]
    if mask_id is not None:
        command.extend(["--predictive-mask-id", mask_id])
    return command


def run_client_wave(
    library: dict[str, Any], split: str, tasks: list[str], mask_ids: list[str], root: Path
) -> None:
    pending = list(mask_ids)
    attempts = {identifier: 0 for identifier in mask_ids}
    max_clients = int(PROTOCOL["runtime"]["max_clients"])
    while pending:
        batch, pending = pending[:max_clients], pending[max_clients:]
        processes = []
        for identifier in batch:
            path = root / split / f"{identifier}.jsonl"
            try:
                verify_client_file(path, tasks, mask_id=identifier, library=library)
                continue
            except (FileNotFoundError, RuntimeError, ValueError):
                pass
            attempts[identifier] += 1
            log_path = ROOT / "logs" / f"client_{root.name}_{split}_{identifier}.log"
            handle = log_path.open("a", buffering=1)
            process = subprocess.Popen(
                client_command(tasks, path, mask_id=identifier), cwd=REPO,
                env=clean_client_env(), stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((identifier, path, process, handle))
        retry = []
        for identifier, path, process, handle in processes:
            return_code = process.wait()
            handle.close()
            try:
                if return_code != 0:
                    raise RuntimeError(f"client exited {return_code}")
                verify_client_file(path, tasks, mask_id=identifier, library=library)
            except Exception as exc:
                if attempts[identifier] >= 3:
                    raise RuntimeError(
                        f"{split}/{identifier}: failed after three resume attempts: {exc}"
                    ) from exc
                retry.append(identifier)
        pending.extend(retry)
        status()


def run_fp16_client(
    library: dict[str, Any], split: str, tasks: list[str], root: Path
) -> None:
    path = root / split / "fp16.jsonl"
    for attempt in range(1, 4):
        try:
            verify_client_file(path, tasks, mask_id=None, library=library)
            return
        except (FileNotFoundError, RuntimeError, ValueError):
            pass
        log_path = ROOT / "logs" / f"client_{root.name}_{split}_fp16.log"
        with log_path.open("a", buffering=1) as handle:
            completed = subprocess.run(
                client_command(tasks, path, mask_id=None), cwd=REPO,
                env=clean_client_env(), stdout=handle, stderr=subprocess.STDOUT,
            )
        if completed.returncode == 0:
            try:
                verify_client_file(path, tasks, mask_id=None, library=library)
                return
            except Exception:
                pass
        if attempt == 3:
            raise RuntimeError(f"{split}/fp16 failed after three resume attempts")


def offline() -> None:
    library = load_library()
    wait_for_gpu()
    OFFLINE_ROOT.mkdir(parents=True, exist_ok=True)
    calibration_buffer = library["selection_buffer"]["path"]
    for split in SPLITS:
        output = OFFLINE_ROOT / f"scores_{split}.json"
        if output.is_file():
            value = json.loads(output.read_text(encoding="utf-8"))
            if value.get("complete") is True and len(value.get("scores") or {}) == 60:
                print(f"[predictive/coverage] offline {split}: 60/60 reuse", flush=True)
                continue
        calibration = REPO / f"runs/errorfold_v3_15x20/calibration/gr00t/{split}"
        command = [
            str(GROOT_PY), str(REPO / "scripts/tools/gr00t_score_outputimpact_plans.py"),
            "--checkpoint", library["split_sources"][split]["checkpoint"]["path"],
            "--base-full-w4-plan", library["full_w4_plan"]["path"],
            "--pack-dir", library["split_sources"][split]["identity_pack"]["path"],
            "--a8", str(calibration / "a8_scales.npz"),
            "--hessian-w4", library["split_sources"][split]["hessian_w4"]["path"],
            "--activation-mode", "dynamic_a8", "--buffer",
            library["split_sources"][split]["selection_buffer"]["path"],
            "--artifact-calibration-buffer", calibration_buffer,
            "--candidate-manifest", str(LIBRARY), "--n-obs", "48",
            "--noise-rule", "A", "--flow-steps", "4", "--batch-size", "8",
            "--device", "cuda", "--out", str(output),
        ]
        env = clean_server_env()
        env["CUDA_VISIBLE_DEVICES"] = str(GPU)
        log = ROOT / "logs" / f"offline_{split}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", buffering=1) as handle:
            completed = subprocess.run(
                command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
        if completed.returncode != 0:
            raise RuntimeError(f"offline scorer failed for {split}; see {log}")
        value = json.loads(output.read_text(encoding="utf-8"))
        if value.get("complete") is not True or len(value.get("scores") or {}) != 60:
            raise RuntimeError(f"offline scorer incomplete for {split}")
        print(f"[predictive/coverage] offline {split}: 60/60", flush=True)
    combined = OFFLINE_ROOT / "scores_all_splits.json"
    command = [
        str(GROOT_PY), str(REPO / "scripts/tools/combine_gr00t_dpac_predictive_scores.py"),
        "--manifest", str(LIBRARY),
    ]
    for split in SPLITS:
        command.extend(["--score", f"{split}={OFFLINE_ROOT / f'scores_{split}.json'}"])
    command.extend(["--out", str(combined)])
    subprocess.run(command, cwd=REPO, check=True)
    freeze_launch_manifest(combined)


def source_artifacts() -> dict[str, Any]:
    paths = [
        "scripts/quantvla_dpac_predictive_protocol.json",
        "scripts/tools/quantvla_predictive_validity.py",
        "scripts/tools/prepare_gr00t_dpac_predictive_validity.py",
        "scripts/tools/gr00t_score_outputimpact_plans.py",
        "scripts/tools/combine_gr00t_dpac_predictive_scores.py",
        "scripts/inference_service.py",
        "scripts/run_robocasa365_gr00t_eval.py",
        "scripts/tools/run_gr00t_dpac_predictive_validity.py",
        "scripts/tools/aggregate_gr00t_dpac_predictive_validity.py",
        "scripts/tools/audit_gr00t_predictive_routes.py",
        "scripts/tools/test_gr00t_dpac_predictive_validity.py",
    ]
    result = {}
    for relative in paths:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = artifact(path)
    return result


def freeze_launch_manifest(offline_path: Path | None = None) -> Path:
    if offline_path is None:
        offline_path = OFFLINE_ROOT / "scores_all_splits.json"
    if not offline_path.is_file():
        raise FileNotFoundError("offline scores must be complete before closed loop")
    payload = {
        "schema_version": 1,
        "kind": "gr00t_dpac_predictive_closed_loop_launch",
        "predictive_validity_protocol": protocol_attestation(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mask_library": artifact(LIBRARY),
        "offline_scores": artifact(offline_path),
        "offline_scores_mtime_ns": offline_path.stat().st_mtime_ns,
        "offline_metrics_frozen_before_closed_loop": True,
        "gpu": GPU,
        "max_egl_clients": int(PROTOCOL["runtime"]["max_clients"]),
        "source_files": source_artifacts(),
    }
    path = ROOT / "closed_loop_launch.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_keys = set(payload) - {"created_utc"}
        if any(existing.get(key) != payload.get(key) for key in comparable_keys):
            raise ValueError("closed-loop launch provenance drift")
        return path
    atomic_json(path, payload)
    return path


def run_closed_loop(*, smoke: bool) -> None:
    library = load_library()
    freeze_launch_manifest()
    mask_ids = ["k01_m00", "k16_m00"] if smoke else [
        row["candidate_id"] for row in library["candidates"]
    ]
    mask_root = SMOKE_ROOT / "masks" if smoke else CLOSED_ROOT / "masks"
    fp16_root = SMOKE_ROOT / "fp16" if smoke else CLOSED_ROOT / "fp16"
    smoke_tasks = set(PROTOCOL["execution"]["smoke_tasks"])
    for split in SPLITS:
        tasks = list(library["closed_loop"]["tasks"][split])
        if smoke:
            tasks = [task for task in tasks if task in smoke_tasks]
            if len(tasks) != 1:
                raise RuntimeError(f"smoke task inventory drift for {split}: {tasks}")
        wait_for_gpu()
        server, handle = start_server(
            library, split, predictive=True,
            label=f"{'smoke' if smoke else 'full'}_{split}_predictive",
        )
        try:
            run_client_wave(library, split, tasks, mask_ids, mask_root)
        finally:
            stop_process(server)
            handle.close()
        wait_for_gpu()
        server, handle = start_server(
            library, split, predictive=False,
            label=f"{'smoke' if smoke else 'full'}_{split}_fp16",
        )
        try:
            run_fp16_client(library, split, tasks, fp16_root)
        finally:
            stop_process(server)
            handle.close()
        status()


def status() -> None:
    library = load_library()
    expected_tasks = {
        task for tasks in library["closed_loop"]["tasks"].values() for task in tasks
    }
    mask_pairs = set()
    for path in (CLOSED_ROOT / "masks").rglob("*.jsonl") if (CLOSED_ROOT / "masks").is_dir() else []:
        for row in rows_in(path):
            if row.get("status") == "complete":
                mask_pairs.add((row.get("predictive_mask_id"), row.get("task")))
    fp16_tasks = set()
    for path in (CLOSED_ROOT / "fp16").rglob("*.jsonl") if (CLOSED_ROOT / "fp16").is_dir() else []:
        for row in rows_in(path):
            if row.get("status") == "complete":
                fp16_tasks.add(row.get("task"))
    complete_masks = sum(
        all((row["candidate_id"], task) in mask_pairs for task in expected_tasks)
        for row in library["candidates"]
    )
    offline_splits = 0
    for split in SPLITS:
        path = OFFLINE_ROOT / f"scores_{split}.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            offline_splits += int(
                value.get("complete") is True and len(value.get("scores") or {}) == 60
            )
    print(
        json.dumps(
            {
                "offline_splits": f"{offline_splits}/3",
                "mask_episode_coverage": f"{len(mask_pairs)}/3000",
                "fully_covered_masks": f"{complete_masks}/60",
                "fp16_episode_coverage": f"{len(fp16_tasks)}/50",
                "outcomes_withheld_until_coverage_gate": True,
            }, indent=2,
        ),
        flush=True,
    )


def aggregate() -> None:
    freeze_launch_manifest()
    command = [
        str(GROOT_PY), str(REPO / "scripts/tools/aggregate_gr00t_dpac_predictive_validity.py"),
        "--manifest", str(LIBRARY),
        "--offline-scores", str(OFFLINE_ROOT / "scores_all_splits.json"),
        "--mask-results-root", str(CLOSED_ROOT / "masks"),
        "--fp16-results-root", str(CLOSED_ROOT / "fp16"),
        "--out-root", str(REPORT_ROOT),
    ]
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("offline", "smoke", "run", "status", "aggregate", "all")
    )
    args = parser.parse_args()
    if args.stage == "offline":
        offline()
    elif args.stage == "smoke":
        run_closed_loop(smoke=True)
    elif args.stage == "run":
        run_closed_loop(smoke=False)
    elif args.stage == "status":
        status()
    elif args.stage == "aggregate":
        aggregate()
    else:
        offline()
        run_closed_loop(smoke=True)
        run_closed_loop(smoke=False)
        aggregate()


if __name__ == "__main__":
    main()
