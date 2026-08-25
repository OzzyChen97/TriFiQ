from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "docs/gdsq_vla_cvpr2026/experiment_registry.json"
MANIFEST = (
    REPO_ROOT
    / "runs/gdsq_week1_preregistered_v1/pi05_selector_official50/manifest.json"
)
SELECTOR_SHA256 = (
    "0f3178726c2b784898f18dfde248d9a9bae152da6ffcfdc299e8bdc02f0bd871"
)
RULE_NAME = "v8_no_oracle_absolute_mechanism_gate_aligned_runtime_rule"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_frozen_before_results_and_covers_exact_matrix() -> None:
    registry = load(REGISTRY)
    record = registry["experiments"]["pi05_runtime_selector_official50"]
    manifest = load(MANIFEST)

    assert sha256_file(MANIFEST) == record["manifest"]["sha256"]
    assert manifest["kind"] == "pi05_gdsq_vla_v8_selector_official50"
    assert manifest["immutable"] is True
    assert manifest["result_blind"] is True
    assert manifest["frozen_before_official_rollouts"] is True

    protocol = manifest["protocol"]
    assert sum(map(len, protocol["task_sets"].values())) == 50
    assert protocol["trial_seeds"] == list(range(50))
    assert protocol["flow_steps"] == 4
    assert protocol["n_action_steps"] == 16
    assert protocol["paired_action_noise"] is True
    assert protocol["statistics"]["bootstrap_draws"] == 10_000

    schedule = manifest["schedule"]
    assert schedule["servers"] == len(manifest["servers"]) == 5
    assert schedule["workers"] == len(manifest["workers"]) == 10
    assert schedule["coverage"] == {
        "covered_episode_keys": 2500,
        "duplicate_episode_keys": 0,
        "expected_episode_keys": 2500,
        "keyset_sha256": "2f6a82c20d6d54225f5ecfc5387b2e82dd517a9fd02fc5dcd7a761131e94ecf6",
    }
    assert len({server["gpu"] for server in manifest["servers"]}) == 5
    assert len({server["port"] for server in manifest["servers"]}) == 5


def test_selector_attestation_is_model_level_and_never_combined() -> None:
    manifest = load(MANIFEST)
    selector = manifest["selector"]
    assert selector["sha256"] == SELECTOR_SHA256
    assert selector["rule_name"] == RULE_NAME
    assert selector["model_id"] == "pi05"
    assert selector["selection_scope"] == "model_level_absolute_mechanism_gate"
    assert selector["selected_variant"] == "ohb"
    assert selector["selected_config_id"] == "gdsq_vla_ohb_only"
    assert selector["uses_task_metadata_for_selection"] is False
    assert selector["rollout_labels_used"] is False
    assert selector["runtime_success_feedback_used"] is False
    assert selector["atmohb_output_allowed"] is False

    for server in manifest["servers"]:
        attestation = server["runtime"]["runtime_selector"]
        decision = attestation["model_decision"]
        assert attestation["selector_sha256"] == SELECTOR_SHA256
        assert attestation["rule_name"] == RULE_NAME
        assert attestation["model_id"] == "pi05"
        assert attestation["uses_task_metadata_for_selection"] is False
        assert decision["selected_variant"] == "ohb"
        assert decision["selected_config_id"] == "gdsq_vla_ohb_only"


def test_preflight_has_exact_2task_by_2seed_attested_coverage() -> None:
    manifest = load(MANIFEST)
    audit_record = manifest["preflight"]["audit"]
    audit_path = Path(audit_record["path"])
    audit = load(audit_path)

    assert sha256_file(audit_path) == audit_record["sha256"]
    assert audit["valid"] is True
    assert audit["diagnostic_only"] is True
    assert audit["paper_claim_enabled"] is False
    assert audit["tasks"] == ["OpenDrawer", "TurnOnMicrowave"]
    assert audit["seeds"] == [0, 1]
    assert audit["expected_episodes"] == audit["observed_episodes"] == 4
    assert audit["missing_episodes"] == audit["duplicate_episodes"] == 0
    assert audit["selector_sha256"] == SELECTOR_SHA256
    assert audit["selector_rule_name"] == RULE_NAME
    assert audit["selected_variant"] == "ohb"
    assert audit["selected_config_id"] == "gdsq_vla_ohb_only"
