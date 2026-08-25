#!/usr/bin/env python3
"""Lightweight tests for the ATM/OHB dynamic selector toolchain."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

features_mod = load_module("atmohb_dynamic_features")
fit_mod = load_module("fit_atmohb_dynamic_selector")
spec_mod = load_module("make_atmohb_dynamic_specs")
agg_mod = load_module("aggregate_atmohb_dynamic_selector")

TASKS = [
    "CoffeeSetupMug",
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
    "CloseFridge",
    "OpenDrawer",
]


def make_summary(prefix: str) -> dict:
    if prefix == "gr00t":
        configs = {
            "fp16": {},
            "w4a8_atmohb": {},
            "cscka_final": {},
            "cscka_final_atm": {},
            "cscka_final_ohb": {},
            "cscka_final_atmohb": {},
        }
    else:
        configs = {
            "fp16": {},
            "quantvla_w4a8_atmohb": {},
            "gdsq_vla": {},
            "gdsq_vla_atm_only": {},
            "gdsq_vla_ohb_only": {},
            "gdsq_vla_atmohb": {},
        }
    for index, task in enumerate(TASKS):
        baseline = 0.55 + 0.03 * index
        for config_id in configs:
            if config_id in {"fp16"}:
                sr = baseline + 0.02
            elif config_id in {"w4a8_atmohb", "quantvla_w4a8_atmohb"}:
                sr = baseline - 0.04
            elif "atm_only" in config_id or config_id.endswith("_atm"):
                sr = baseline + (0.05 if "PickPlace" in task else -0.01)
            elif "ohb_only" in config_id or config_id.endswith("_ohb"):
                sr = baseline + (0.04 if task.startswith("Open") or task.startswith("Close") else 0.0)
            elif "atmohb" in config_id:
                sr = baseline + 0.01
            else:
                sr = baseline
            configs[config_id].setdefault("per_task", {})[task] = {"sr": round(sr, 4), "successes": int(sr * 50), "episodes": 50}
    return {"tasks": TASKS, "seeds": list(range(50)), "configs": configs}


def test_deterministic_and_dev_only() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gr = root / "gr.json"
        pi = root / "pi.json"
        gr.write_text(json.dumps(make_summary("gr00t")), encoding="utf-8")
        pi.write_text(json.dumps(make_summary("pi05")), encoding="utf-8")
        features = features_mod.build_features(gr, pi)
        selector_a = fit_mod.build_selector(features)
        selector_b = fit_mod.build_selector(features)
        assert selector_a == selector_b
        for model in selector_a["models"].values():
            assert model["fit"]["development_task_count"] == 4
            assert model["fit"]["heldout_tasks_used_for_thresholds"] == []
            assert set(model["fit"]["development_tasks_used"]) == set(features["development_tasks"])
        # Held-out analysis-only labels can change without changing thresholds/normalizers.
        changed = json.loads(json.dumps(features))
        for model in changed["models"].values():
            for task, row in model["tasks"].items():
                if row["features"]["is_heldout"]:
                    row["analysis_only"]["retrospective_best_variant"] = "atmohb"
        selector_c = fit_mod.build_selector(changed)
        for model_id in selector_a["models"]:
            assert selector_a["models"][model_id]["fit"] == selector_c["models"][model_id]["fit"]


def test_no_oracle_guard() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gr = root / "gr.json"
        pi = root / "pi.json"
        gr.write_text(json.dumps(make_summary("gr00t")), encoding="utf-8")
        pi.write_text(json.dumps(make_summary("pi05")), encoding="utf-8")
        selector = fit_mod.build_selector(features_mod.build_features(gr, pi))
        selector["no_oracle_contract"]["analysis_only_derived_threshold_flag"] = True
        try:
            agg_mod.build_replay(selector, {"gr00t": make_summary("gr00t"), "pi05": make_summary("pi05")})
        except ValueError as exc:
            assert "analysis_only" in str(exc)
        else:
            raise AssertionError("no-oracle guard did not fire")


def test_spec_grouping() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gr = root / "gr.json"
        pi = root / "pi.json"
        gr.write_text(json.dumps(make_summary("gr00t")), encoding="utf-8")
        pi.write_text(json.dumps(make_summary("pi05")), encoding="utf-8")
        selector = fit_mod.build_selector(features_mod.build_features(gr, pi))
        specs = spec_mod.build_specs(selector, [0, 1, 2], heldout_only=True)
        assert specs["seeds"] == [0, 1, 2]
        assert specs["heldout_only"] is True
        assert specs["models"]["gr00t"]["config_id_mapping"]["baseline"] == "cscka_final"
        assert specs["models"]["pi05"]["config_id_mapping"]["atmohb"] == "gdsq_vla_atmohb"
        for model in specs["models"].values():
            grouped_tasks = [task for group in model["groups"].values() for task in group["tasks"]]
            assert grouped_tasks
            assert all(task not in features_mod.DEVELOPMENT_TASKS for task in grouped_tasks)


def test_v4_mechanism_dominant_no_oracle() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gr = root / "gr.json"
        pi = root / "pi.json"
        gr.write_text(json.dumps(make_summary("gr00t")), encoding="utf-8")
        pi.write_text(json.dumps(make_summary("pi05")), encoding="utf-8")
        features = features_mod.build_features(gr, pi)

        gr_plan = root / "gr_plan.json"
        pi_plan = root / "pi_plan.json"
        gr_plan.write_text(
            json.dumps(
                {
                    "layers": {
                        "action_head.model.transformer_blocks.0.ff.net.0.proj": {"bits": 4, "skip": False},
                        "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj": {"bits": 4, "skip": False},
                    }
                }
            ),
            encoding="utf-8",
        )
        pi_plan.write_text(
            json.dumps(
                {
                    "layers": {
                        "paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj": {"bits": 4, "skip": False},
                        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj": {"bits": 4, "skip": False},
                    }
                }
            ),
            encoding="utf-8",
        )
        gr_atmohb = root / "gr_atmohb.json"
        pi_atmohb = root / "pi_atmohb.json"
        gr_atmohb.write_text(
            json.dumps(
                {
                    "layer": {
                        "all": [0.90, 0.92, 0.94, 1.0],
                        "beta_perhead": [0.90, 1.0, 1.10, 1.0],
                    }
                }
            ),
            encoding="utf-8",
        )
        pi_atmohb.write_text(
            json.dumps(
                {
                    "layers": {
                        "layer": {
                            "all": [0.99, 1.0, 1.01, 1.0],
                            "beta_perhead": [0.75, 0.80, 0.85, 0.90],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        selector = fit_mod.build_selector_v4(
            features,
            gr00t_plan=gr_plan,
            gr00t_atmohb=gr_atmohb,
            pi05_plan=pi_plan,
            pi05_atmohb=pi_atmohb,
        )
        assert selector["rule_name"].startswith("v4_")
        assert set(selector["models"]["gr00t"]["selected"].values()) == {"atm"}
        assert set(selector["models"]["pi05"]["selected"].values()) == {"ohb"}
        assert selector["models"]["gr00t"]["fit"]["rollout_labels_used"] is False
        assert selector["models"]["pi05"]["fit"]["task_metadata_used_for_variant_selection"] is False
        assert selector["fit_metadata"]["atmohb_combination_disabled"] is True

        changed = json.loads(json.dumps(features))
        for model in changed["models"].values():
            for row in model["tasks"].values():
                row["analysis_only"]["retrospective_best_variant"] = "atmohb"
        selector_changed = fit_mod.build_selector_v4(
            changed,
            gr00t_plan=gr_plan,
            gr00t_atmohb=gr_atmohb,
            pi05_plan=pi_plan,
            pi05_atmohb=pi_atmohb,
        )
        assert selector["models"] == selector_changed["models"]


def test_v7_quick_sr_refinement() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        tasks = ["CloseFridge", "OpenDrawer"]
        seeds = [0, 1]
        base_selector = {
            "schema_version": 5,
            "rule_name": "v6_test_aligned_rule",
            "fit_metadata": {
                "cross_model_quantization_config_aligned": True,
                "deployment": {"shared_quantization_contract": {"logical_profile": "gdsq_vla"}},
            },
            "models": {
                "gr00t": {
                    "baseline_config_id": "cscka_final",
                    "variant_config_ids": {
                        "baseline": "cscka_final",
                        "atm": "cscka_final_atm",
                        "ohb": "cscka_final_ohb",
                        "atmohb": "cscka_final_atmohb",
                    },
                    "fit": {
                        "mechanism_profile": {
                            "dominant_variant": "atm",
                            "decision_rule": "test ATM mechanism rule",
                        }
                    },
                    "tasks": {
                        task: {"selected_variant": "atm", "selected_config_id": "cscka_final_atm"}
                        for task in tasks
                    },
                    "selected": {task: "atm" for task in tasks},
                },
                "pi05": {
                    "baseline_config_id": "gdsq_vla",
                    "variant_config_ids": {
                        "baseline": "gdsq_vla",
                        "atm": "gdsq_vla_atm_only",
                        "ohb": "gdsq_vla_ohb_only",
                        "atmohb": "gdsq_vla_atmohb",
                    },
                    "fit": {
                        "mechanism_profile": {
                            "dominant_variant": "ohb",
                            "decision_rule": "test OHB mechanism rule",
                        }
                    },
                    "tasks": {
                        task: {"selected_variant": "ohb", "selected_config_id": "gdsq_vla_ohb_only"}
                        for task in tasks
                    },
                    "selected": {task: "ohb" for task in tasks},
                },
            },
        }
        base_path = root / "base_selector.json"
        base_path.write_text(json.dumps(base_selector), encoding="utf-8")

        def write_rows(path: Path, config_id: str, successes: dict[tuple[str, int], bool]) -> None:
            path.mkdir(parents=True, exist_ok=True)
            rows = []
            for task in tasks:
                for seed in seeds:
                    rows.append(
                        {
                            "config": config_id,
                            "task": task,
                            "seed": seed,
                            "success": successes[(task, seed)],
                            "status": "complete",
                            "task_set": "atomic_seen",
                            "split": "target",
                            "n_action_steps": 16,
                            "replan_steps": 16,
                            "flow_steps": 4,
                            "paired_action_noise": True,
                            "action_noise_protocol": fit_mod.PAIRED_NOISE_PROTOCOL,
                        }
                    )
            (path / "rows.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

        roots = {
            "gr00t": {"baseline": root / "gr_base", "stabilizer": root / "gr_atm"},
            "pi05": {"baseline": root / "pi_base", "stabilizer": root / "pi_ohb"},
        }
        baseline_outcomes = {
            ("CloseFridge", 0): True,
            ("CloseFridge", 1): False,
            ("OpenDrawer", 0): True,
            ("OpenDrawer", 1): False,
        }
        gr_atm_outcomes = dict(baseline_outcomes)
        gr_atm_outcomes[("CloseFridge", 1)] = True
        pi_ohb_outcomes = dict(baseline_outcomes)
        write_rows(roots["gr00t"]["baseline"], "cscka_final", baseline_outcomes)
        write_rows(roots["gr00t"]["stabilizer"], "cscka_final_atm", gr_atm_outcomes)
        write_rows(roots["pi05"]["baseline"], "gdsq_vla", baseline_outcomes)
        write_rows(roots["pi05"]["stabilizer"], "gdsq_vla_ohb_only", pi_ohb_outcomes)

        selector = fit_mod.build_selector_v7(
            base_selector,
            base_selector_path=base_path,
            quick_tasks=tasks,
            quick_seeds=seeds,
            result_roots=roots,
        )
        assert selector["models"]["gr00t"]["selected"]["CloseFridge"] == "atm"
        assert selector["models"]["gr00t"]["selected"]["OpenDrawer"] == "baseline"
        assert set(selector["models"]["pi05"]["selected"].values()) == {"baseline"}
        assert selector["no_oracle_contract"]["benchmark_tuned"] is True
        assert selector["fit_metadata"]["cross_model_quantization_config_aligned"] is True


def main() -> None:
    test_deterministic_and_dev_only()
    test_no_oracle_guard()
    test_spec_grouping()
    test_v4_mechanism_dominant_no_oracle()
    test_v7_quick_sr_refinement()
    print("atmohb dynamic selector tests passed")


if __name__ == "__main__":
    main()