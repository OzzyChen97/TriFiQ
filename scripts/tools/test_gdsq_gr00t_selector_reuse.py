from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("gdsq_gr00t_selector_reuse_attestation.py")
SPEC = importlib.util.spec_from_file_location("gdsq_gr00t_selector_reuse", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reuse_attestation_covers_every_static_episode() -> None:
    rows_text, summary = MODULE.build()
    rows = [json.loads(line) for line in rows_text.splitlines() if line.strip()]
    assert summary["valid"] is True
    assert summary["raw_results_modified"] is False
    assert summary["coverage"] == {
        "tasks": 50,
        "seeds_per_task": 50,
        "expected_episodes": 2500,
        "observed_episodes": 2500,
        "missing_episodes": 0,
        "duplicate_episodes": 0,
    }
    assert len(rows) == 2500
    assert len({(row["task_set"], row["task"], row["seed"]) for row in rows}) == 2500


def test_every_reused_row_carries_frozen_selector_contract() -> None:
    rows_text, _ = MODULE.build()
    for row in map(json.loads, rows_text.splitlines()):
        assert row["runtime_selector_enabled"] is True
        assert row["selector_sha256"] == MODULE.SELECTOR_SHA256
        assert row["selector_rule_name"] == MODULE.SELECTOR_RULE
        assert row["selector_model_id"] == "gr00t"
        assert row["selected_variant"] == "baseline"
        assert row["selected_config_id"] == MODULE.CONFIG
        assert row["reuse_static_result"] is True
