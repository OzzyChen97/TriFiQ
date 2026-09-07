#!/usr/bin/env python3
"""Run frozen projection comparisons on GPUs 0-3 with bounded concurrency.

Requires all four production-path parity reports. Ramps one job per GPU at a
 time; admission accounts for server + three simulators + 2 GiB reserve.
Never changes a mask, seed, threshold or outcome-dependent experiment choice.
Failures stop new launches; existing jobs can finish and results are retained.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/budget_projection_comparison_v1"
GPUS = (0, 1, 2, 3)
SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
GROOT_PY = "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
SIM_PY = "/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CONFIG = "full_context_w4a8_dynamic_profile"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"Immutable execution drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pi05_semantic_metadata_hash(metadata):
    """Validate and return the server's stable semantic runtime identity."""
    runtime = metadata.get("openpi_runtime") or {}
    claimed = runtime.get("semantic_metadata_sha256")
    require(bool(claimed), "Live pi05 semantic metadata hash missing")
    stable = json.loads(json.dumps(metadata, sort_keys=True))
    stable_runtime = stable.get("openpi_runtime") or {}
    stable_runtime.pop("gpu_memory_bytes", None)
    stable_runtime.pop("semantic_metadata_sha256", None)
    computed = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(claimed == computed, f"Live pi05 semantic metadata hash mismatch: {claimed} != {computed}")
    return claimed


def parity_reports():
    paths = [OUT / "parity" / f"gr00t_{split}.json" for split in SPLITS]
    paths.append(OUT / "parity/pi05_atomic_seen.json")
    for path in paths:
        if not path.exists():
            return None
        d = read(path)
        if d.get("failure"):
            raise RuntimeError(f"Parity failed: {path}: {d['failure']}")
        if not d.get("complete"):
            return None
        require(d["manifest_sha256"] == digest(OUT / "manifest.json"), "Parity manifest drift")
        require(d["code_sha256"] == digest(ROOT / "scripts/tools/check_budget_projection_parity.py"), "Parity code drift")
        require(len(d["checks"]) == 7, "Incomplete parity candidate coverage")
        require(all(r["passed"] and r["packed_residency"]["packed_low_bit_residency"] for r in d["checks"].values()), "Nonproduction parity")
    return [{"path":str(p), "sha256":digest(p)} for p in paths]


def make_execution(parity):
    m = read(OUT / "manifest.json")
    jobs = []
    # Finish GR00T before pi0.5; this order never depends on success rates.
    for model in ("gr00t", "pi05"):
        for split in SPLITS:
            for identifier, row in m["models"][model]["plans"].items():
                if identifier != row["execution_alias"]:
                    continue
                require(digest(row["path"]) == row["sha256"], "Candidate drift")
                plan = read(row["path"])
                subset_split = split if model == "gr00t" else "atomic_seen"
                weights = OUT / "hessian" / identifier / subset_split / "hessian_w4.npz"
                meta = read(str(weights) + ".json")
                require(meta["deployment_plan_sha256"] == row["sha256"], "Subset plan mismatch")
                require(meta["npz_sha256"] == digest(weights), "Subset content drift")
                jobs.append({"id":f"{identifier}__{split}","model":model,"candidate":identifier,"split":split,
                             "plan":row["path"],"plan_sha256":row["sha256"],"hessian":str(weights),
                             "hessian_sha256":meta["npz_sha256"],"wrapped":plan["quantized_w4_layers"],
                             "tasks":m["tasks"][split],"seeds":m["protocol"]["evaluation"]["seeds"]})
    files = [Path(__file__), ROOT/"scripts/tools/run_robocasa_atomic_matrix.py",
             ROOT/"scripts/tools/run_crit4_trial_driver.py", ROOT/"scripts/run_robocasa365_gr00t_eval.py",
             ROOT/"scripts/inference_service.py", ROOT/"scripts/run_robocasa365_quant_serve.sh",
             ROOT/"scripts/run_pi05_formal_server.sh", ROOT/"scripts/run_pi05_formal_worker_seeded.sh",
             ROOT/"scripts/run_robocasa365_pi05_eval.py", ROOT/"code/pi05/openpi/scripts/serve_pi05_quant_policy.py"]
    result = {"manifest_sha256":digest(OUT/"manifest.json"),"parity":parity,
              "authorization":"User authorized terminating old GPU processes and concurrent use of GPU0,1,2,3",
              "allowed_gpus":list(GPUS),"max_slots_per_gpu":3,"clients_per_slot":3,"ramp_seconds":90,
              "server_budget_mib":{"gr00t":8000,"pi05":11000},"client_budget_mib":2000,"reserve_mib":2048,
              "operational_override":"Preparation-only gpu_launch_enabled=false superseded by explicit user authorization; statistical protocol unchanged",
              "code":[{"path":str(p),"sha256":digest(p)} for p in files],"jobs":jobs}
    save(OUT/"execution_manifest.json",result,immutable=True)
    return result


def environment():
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES",None)
    env.update({"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
                "JAX_PLATFORMS":"cpu","PYTHONUNBUFFERED":"1"})
    return env


def completed_keys(job):
    directory = OUT/"rollouts"/job["id"]
    paths = sorted(directory.glob(f"{job['candidate']}_s*.jsonl")) if job["model"] == "gr00t" else sorted((directory/"results"/CONFIG).glob("*.jsonl"))
    seen = set()
    expected = {(t,s) for t in job["tasks"] for s in job["seeds"]}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row= json.loads(line)
            require(row.get("status") == "complete", f"Invalid episode status: {path}")
            require(row.get("config") == (job["candidate"] if job["model"] == "gr00t" else CONFIG), "Unexpected result configuration")
            key = (row["task"],int(row["seed"]))
            require(key in expected and key not in seen, f"Duplicate or out-of-protocol key: {path} {key}")
            require(row.get("paired_action_noise") is True, "Unpaired action noise")
            seen.add(key)
    return seen, expected


def worker(job, gpu, port):
    require(gpu in GPUS,"GPU outside allowlist")
    require(digest(job["plan"]) == job["plan_sha256"], "Plan drift")
    run = OUT/"rollouts"/job["id"]
    run.mkdir(parents=True,exist_ok=True)
    env = environment()
    if job["model"] == "gr00t":
        calibration=ROOT/f"runs/archive_previous_versions/errorfold_v3_15x20/calibration/gr00t/{job['split']}"
        config={"id":job["candidate"],"gpu":gpu,"port":port,"plan":job["plan"],"packdir":str(calibration/"identity_pack"),
                "hessian_w4":job["hessian"],"activation_mode":"dynamic_a8","act_scale":None,"expected_wrapped":job["wrapped"],
                "atm":None,"ohb":False,"ohb_only":False,"errorfold":None}
        spec={"kind":"independent_projection_execution","configs":[config]}
        specpath=OUT/"execution_specs"/(job["id"]+".json")
        save(specpath,spec,immutable=True)
        tasks=",".join(job["tasks"])
        command=[GROOT_PY,str(ROOT/"scripts/tools/run_robocasa_atomic_matrix.py"),"--spec",str(specpath),"--run-dir",str(run),
                 "--phase","dev","--seeds",",".join(map(str,job["seeds"])),
                 "--checkpoint",str(ROOT/f"checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/{job['split']}/checkpoint-60000"),
                 "--task-set",job["split"],"--tasks",tasks,"--dev-tasks",tasks,"--n-shards","6","--seed-shards-per-task","1",
                 "--egl-device-pool",str(gpu),"--trial-batch-size","5","--max-concurrent-clients","3","--action-noise","paired",
                 "--diagnostic-only","--allow-shared-gpus","--server-memory-budget-mib","8000","--client-memory-budget-mib","2000",
                 "--gpu-reserve-mib","2048","--concurrent-client-memory"]
        subprocess.run(command,cwd=ROOT,env=env,check=True)
    else:
        instance=job["id"]
        control=OUT/"pi05_control"
        env.update({"PI05_CONTROL_DIR":str(control),"PI05_FULL_CONTEXT_PLAN":job["plan"],
                    "PI05_FULL_CONTEXT_HESSIAN_W4":job["hessian"],
                    "PI05_FULL_CONTEXT_PACK_DIR":str(ROOT/"runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/identity_pack"),
                    "PI05_FLOW_STEPS":"4","PI05_TASK_SETS":job["split"]})
        server=ROOT/"scripts/run_pi05_formal_server.sh"
        children=[]
        try:
            subprocess.run(["bash",str(server),"start",CONFIG,str(gpu),str(port),instance],cwd=ROOT,env=env,check=True)
            metadata=read(control/f"{instance}.runtime.json")
            rt=metadata["openpi_runtime"];q=rt["duquant"]
            require(q["plan_sha256"]==job["plan_sha256"] and q["hessian_w4_sha256"]==job["hessian_sha256"],"Live pi05 artifact mismatch")
            require(q["wrapped_layers"]==job["wrapped"] and q["packed_low_bit_residency"] is True,"Live pi05 quantization mismatch")
            metadata_hash=pi05_semantic_metadata_hash(metadata)
            save(run/"runtime_attestation.json",{"job":job,"gpu":gpu,"port":port,"metadata_sha256":metadata_hash,"metadata":metadata},immutable=True)
            for shard in range(3):
                handle=open(run/f"client_{shard}.log","a")
                command=["bash",str(ROOT/"scripts/run_pi05_formal_worker_seeded.sh"),CONFIG,str(port),str(gpu),str(shard),"3",
                         f"{job['seeds'][0]}-{job['seeds'][-1]}",f"projection_{shard}",metadata_hash,str(run)]
                children.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
                handle.close()
            for child in children:
                require(child.wait()==0,"pi05 client failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            subprocess.run(["bash",str(server),"stop",instance],cwd=ROOT,env=env,check=False)
    seen,expected=completed_keys(job)
    require(seen==expected,f"Incomplete job {job['id']}: {len(seen)}/{len(expected)}")
    save(OUT/"completed"/(job["id"]+".json"),{"job":job["id"],"episodes":len(seen),"complete":True})
    print("JOB_COMPLETE",job["id"],len(seen),flush=True)


def gpu_free():
    text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True)
    return {int(a):float(b) for a,b in (line.split(",") for line in text.splitlines())}


def run():
    lock=open(OUT/"scheduler.lock","a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time()
    while True:
        parity=parity_reports()
        if parity is not None:
            break
        require(time.time()-start<3600,"Parity timeout; inspect logs before retrying")
        save(OUT/"scheduler_status.json",{"phase":"waiting_for_parity","pid":os.getpid(),"gpus":list(GPUS)})
        time.sleep(15)
    execution=make_execution(parity)
    active={};finished=set();failed=[];last={g:0 for g in GPUS}
    for model in ("gr00t","pi05"):
        pending=[]
        for job in execution["jobs"]:
            if job["model"]!=model:
                continue
            seen,expected=completed_keys(job)
            if seen==expected:
                finished.add(job["id"])
            else:
                pending.append(job)
        while pending or active:
            for identifier,(process,job,gpu,handle) in list(active.items()):
                rc=process.poll()
                if rc is None:
                    continue
                handle.close(); del active[identifier]
                if rc:
                    failed.append({"job":identifier,"exit_code":rc})
                else:
                    finished.add(identifier)
            if not failed and pending:
                free=gpu_free()
                for gpu in GPUS:
                    if not pending:
                        break
                    slots=[x for x in active.values() if x[2]==gpu]
                    needed=execution["server_budget_mib"][model]+3*2000+2048
                    if len(slots)>=3 or time.time()-last[gpu]<90 or free.get(gpu,0)<needed:
                        continue
                    # Do not overlap a new model load with an unready previous load.
                    if any(not ((OUT/"rollouts"/j["id"]/"runtime_info.json").exists() if model=="gr00t" else (OUT/"rollouts"/j["id"]/"runtime_attestation.json").exists()) for _,j,_,_ in slots):
                        continue
                    job=pending.pop(0)
                    index=next(i for i,j in enumerate(execution["jobs"]) if j["id"]==job["id"])
                    port=24000+index
                    handle=open(OUT/"logs"/(job["id"]+".log"),"a")
                    command=[sys.executable,str(Path(__file__).resolve()),"worker","--job",job["id"],"--gpu",str(gpu),"--port",str(port)]
                    process=subprocess.Popen(command,cwd=ROOT,env=environment(),stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    active[job["id"]]=(process,job,gpu,handle);last[gpu]=time.time()
                    print("JOB_STARTED",job["id"],"gpu",gpu,"pid",process.pid,flush=True)
            save(OUT/"scheduler_status.json",{"phase":model,"pid":os.getpid(),"active":[{"job":i,"pid":p.pid,"gpu":g} for i,(p,j,g,h) in active.items()],
                                              "pending":[j["id"] for j in pending],"finished":sorted(finished),"failed":failed,"gpus":list(GPUS)})
            if failed and not active:
                raise RuntimeError(f"Jobs failed; no new work launched: {failed}")
            time.sleep(15)
    save(OUT/"scheduler_status.json",{"phase":"complete","finished":sorted(finished),"failed":failed,"gpus":list(GPUS)})
    print("ALL_ROLLOUTS_COMPLETE",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("run","worker"))
    p.add_argument("--job");p.add_argument("--gpu",type=int);p.add_argument("--port",type=int)
    args=p.parse_args()
    if args.mode=="run":
        run()
    else:
        execution=read(OUT/"execution_manifest.json")
        for item in execution["code"]:
            require(digest(item["path"])==item["sha256"],"Execution code changed after freeze")
        worker(next(j for j in execution["jobs"] if j["id"]==args.job),args.gpu,args.port)


if __name__=="__main__":
    main()