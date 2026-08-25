from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest_tool = load_module(
    "pi05_make_week1_manifest",
    REPO_ROOT / "scripts/tools/pi05_make_week1_manifest.py",
)
driver = load_module(
    "run_crit4_trial_driver",
    REPO_ROOT / "scripts/tools/run_crit4_trial_driver.py",
)


def schedule_rows():
    servers = [
        {"instance": f"g{gpu}", "gpu": gpu, "port": 20200 + gpu}
        for gpu in range(3, 8)
    ]
    workers = []
    seed_quarters = (("q0", 0, 12), ("q1", 13, 24), ("q2", 25, 37), ("q3", 38, 49))
    for shard in range(5):
        for quarter, (suffix, start, end) in enumerate(seed_quarters):
            gpu = 3 + ((shard + quarter) % 5)
            workers.append(
                {
                    "worker_id": f"g{gpu}_s{shard}_{suffix}",
                    "server_instance": f"g{gpu}",
                    "task_shard_index": shard,
                    "task_shard_count": 5,
                    "trial_seed_start": start,
                    "trial_seed_end": end,
                }
            )
    return servers, workers


def test_scopes_exclude_only_the_frozen_development_tasks() -> None:
    all50 = manifest_tool.task_sets_for_scope("all50")
    dev4 = manifest_tool.task_sets_for_scope("dev4")
    heldout46 = manifest_tool.task_sets_for_scope("heldout46")
    all_tasks = {task for tasks in all50.values() for task in tasks}
    dev_tasks = {task for tasks in dev4.values() for task in tasks}
    heldout_tasks = {task for tasks in heldout46.values() for task in tasks}

    assert list(dev_tasks) or dev_tasks
    assert dev4 == {"atomic_seen": manifest_tool.DEV4}
    assert len(all_tasks) == 50
    assert len(dev_tasks) == 4
    assert len(heldout_tasks) == 46
    assert dev_tasks == set(manifest_tool.DEV4)
    assert dev_tasks.isdisjoint(heldout_tasks)
    assert dev_tasks | heldout_tasks == all_tasks


def test_five_server_twenty_worker_latin_schedule_has_exact_scope_coverage() -> None:
    servers, workers = schedule_rows()
    assert len(workers) == 20
    assert {gpu: sum(row["server_instance"] == f"g{gpu}" for row in workers) for gpu in range(3, 8)} == {
        gpu: 4 for gpu in range(3, 8)
    }
    expected = {"all50": 2500, "heldout46": 2300, "dev4": 200}
    for scope, count in expected.items():
        coverage = manifest_tool.validate_schedule(
            servers, workers, manifest_tool.task_sets_for_scope(scope)
        )
        assert coverage["expected_episode_keys"] == count
        assert coverage["covered_episode_keys"] == count
        assert coverage["duplicate_episode_keys"] == 0


def test_four_client_dev_and_formal_control_schedules_have_exact_coverage() -> None:
    dev_servers = [{"instance": "dev", "gpu": 3, "port": 20303}]
    dev_workers = []
    for shard in range(2):
        for suffix, start, end in (("lo", 0, 24), ("hi", 25, 49)):
            dev_workers.append(
                {
                    "worker_id": f"dev_s{shard}_{suffix}",
                    "server_instance": "dev",
                    "task_shard_index": shard,
                    "task_shard_count": 2,
                    "trial_seed_start": start,
                    "trial_seed_end": end,
                }
            )
    dev = manifest_tool.validate_schedule(
        dev_servers, dev_workers, manifest_tool.task_sets_for_scope("dev4")
    )
    assert dev["covered_episode_keys"] == 200
    assert dev["duplicate_episode_keys"] == 0

    formal_servers = [
        {"instance": "formal0", "gpu": 3, "port": 20403},
        {"instance": "formal1", "gpu": 4, "port": 20404},
    ]
    formal_workers = []
    quarters = ((0, 12), (13, 24), (25, 37), (38, 49))
    for shard, instance in enumerate(("formal0", "formal1")):
        for quarter, (start, end) in enumerate(quarters):
            formal_workers.append(
                {
                    "worker_id": f"formal_s{shard}_q{quarter}",
                    "server_instance": instance,
                    "task_shard_index": shard,
                    "task_shard_count": 2,
                    "trial_seed_start": start,
                    "trial_seed_end": end,
                }
            )
    formal = manifest_tool.validate_schedule(
        formal_servers,
        formal_workers,
        manifest_tool.task_sets_for_scope("heldout46"),
    )
    assert formal["covered_episode_keys"] == 2300
    assert formal["duplicate_episode_keys"] == 0


def test_no_guards_terminal_crash_is_committed_as_failure() -> None:
    row = driver.terminal_failure_row(
        config="ablation_no_guards",
        manifest_sha="manifest",
        config_sha="config",
        task="OpenDrawer",
        trial=7,
        seed=7,
        paired_action_noise=True,
    )
    assert row["success"] is False
    assert row["formal_failure"] is True
    assert row["crashed"] is False
    assert row["failure_reason"] == "all_retry_attempts_failed"
    assert row["paired_action_noise"] is True
    assert row["steps"] == row["replans"] == 0
