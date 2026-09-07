#!/usr/bin/env python3
"""Aggregate only complete, byte-matched prospective projection results.

Eight paired exact McNemar tests form one Holm family. Task-cluster bootstrap
intervals are descriptive. Reads raw rows without rewriting any outcome.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from aggregate_full_context_table1 import exact_mcnemar
from run_budget_projection_comparison import OUT, CONFIG, completed_keys, read, require, save, digest


def main():
    manifest = read(OUT / "manifest.json")
    execution = read(OUT / "execution_manifest.json")
    require(execution["manifest_sha256"] == digest(OUT / "manifest.json"), "Execution manifest mismatch")
    all_rows = {}
    sources = []
    for job in execution["jobs"]:
        seen, expected = completed_keys(job)
        require(seen == expected, f"Incomplete configuration: {job['id']}")
        directory = OUT / "rollouts" / job["id"]
        if job["model"] == "gr00t":
            runtime = read(directory / "runtime_info.json")[job["candidate"]]
            require(runtime["plan_sha256"] == job["plan_sha256"], "GR00T runtime plan mismatch")
            require(runtime["hessian_w4_sha256"] == job["hessian_sha256"], "GR00T runtime weights mismatch")
            paths = sorted(directory.glob(f"{job['candidate']}_s*.jsonl"))
        else:
            runtime = read(directory / "runtime_attestation.json")
            require(runtime["job"] == job, "pi05 runtime job mismatch")
            paths = sorted((directory / "results" / CONFIG).glob("*.jsonl"))
        rows = all_rows.setdefault(job["candidate"], {})
        for path in paths:
            sources.append({"path":str(path),"sha256":digest(path)})
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                require(isinstance(row.get("success"),bool), "Missing/nonboolean task success")
                key=(row["task"],int(row["seed"]))
                require(key not in rows,"Duplicate across splits")
                rows[key]=row["success"]
    comparisons=[]
    rates={}
    rng=np.random.default_rng(20260906)
    for model, content in manifest["models"].items():
        for identifier,item in content["plans"].items():
            rows=all_rows[item["execution_alias"]]
            require(len(rows)==500,"Expected 500 episodes per configuration")
            rates[identifier]={"episodes":500,"successes":sum(rows.values()),"success_rate":sum(rows.values())/500}
        for budget,condition in content["conditions"].items():
            ours_id=condition["arms"]["full_policy"]["candidate_id"]
            ours=all_rows[content["plans"][ours_id]["execution_alias"]]
            for baseline in ("final_action_error","largest_first"):
                other_id=condition["arms"][baseline]["candidate_id"]
                other=all_rows[content["plans"][other_id]["execution_alias"]]
                require(set(ours)==set(other),"Unpaired comparison keys")
                require(condition["arms"][baseline]["static_bytes"]==condition["matched_actual_bytes"],"Comparison byte mismatch")
                wins=sum(ours[k] and not other[k] for k in ours)
                losses=sum(other[k] and not ours[k] for k in ours)
                tasks=sorted({t for t,s in ours})
                deltas=np.array([np.mean([int(ours[(t,s)])-int(other[(t,s)]) for s in manifest["protocol"]["evaluation"]["seeds"]]) for t in tasks])
                draws=deltas[rng.integers(0,len(tasks),size=(10000,len(tasks)))].mean(axis=1)
                comparisons.append({"model":model,"budget":budget,"baseline":baseline,"full_policy":ours_id,
                                    "control":other_id,"static_bytes":condition["matched_actual_bytes"],
                                    "wins":wins,"losses":losses,"delta":float(deltas.mean()),
                                    "raw_p":exact_mcnemar(wins,losses),"descriptive_cluster_ci95":np.quantile(draws,[.025,.975]).tolist()})
    require(len(comparisons)==8,"Expected fixed eight-test family")
    previous=0.0
    for rank,index in enumerate(sorted(range(8),key=lambda i:comparisons[i]["raw_p"])):
        previous=max(previous,min(1.0,(8-rank)*comparisons[index]["raw_p"]))
        comparisons[index]["holm_p"]=previous
    save(OUT/"aggregate.json",{"complete":True,"kind":"prospective_projection_comparison",
                              "manifest_sha256":digest(OUT/"manifest.json"),"execution_sha256":digest(OUT/"execution_manifest.json"),
                              "aggregator_sha256":digest(Path(__file__)),"sources":sources,"rates":rates,"comparisons":comparisons,
                              "selection_feedback_allowed":False},immutable=True)
    print("AGGREGATE_COMPLETE",OUT/"aggregate.json")


if __name__ == "__main__":
    main()