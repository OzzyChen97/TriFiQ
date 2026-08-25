#!/usr/bin/env python3
"""Configuration classes and constants for the pi0.5 RoboCerebra evaluation.

Ported from github.com/qiuboxiang/RoboCerebra evaluation/config.py with all
OpenVLA-specific fields removed and a pi0.5 policy-server section added.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

BENCH_ROOT_ENV = "ROBOCEREBRA_BENCH_ROOT"
INIT_FILES_ROOT_ENV = "ROBOCEREBRA_INIT_FILES_ROOT"

DEFAULT_TASK_TYPES = [
    "Ideal",
    "Memory_Execution",
    "Memory_Exploration",
    "Mix",
    "Observation_Mismatching",
    "Random_Disturbance",
]


def _read_env(name: str) -> Optional[str]:
    """Return a stripped environment variable or ``None``."""
    value = os.getenv(name)
    if value is None:
        return None

    value = value.strip()
    return value or None


def _default_init_files_root() -> Optional[str]:
    """Resolve the init files root from explicit or derived configuration."""
    explicit_root = _read_env(INIT_FILES_ROOT_ENV)
    if explicit_root:
        return explicit_root

    bench_root = _read_env(BENCH_ROOT_ENV)
    if not bench_root:
        return None

    return str(Path(bench_root) / "init_files")


class TaskSuite(str, Enum):
    ROBOCEREBRA = "robocerebra"


TASK_MAX_STEPS: Dict[TaskSuite, int] = {
    TaskSuite.ROBOCEREBRA: 400,
}

SCENE_MAPPINGS = {
    "COFFEE_TABLESCENE": "libero_coffee_table_manipulation",
    "KITCHEN_TABLESCENE": "libero_kitchen_tabletop_manipulation",
    "STUDY_TABLESCENE": "libero_study_tabletop_manipulation",
}

MOVABLE_OBJECT_LIST = [
    "alphabet_soup",
    "bbq_sauce",
    "butter",
    "chocolate_pudding",
    "cookies",
    "cream_cheese",
    "ketchup",
    "macaroni_and_cheese",
    "milk",
    "orange_juice",
    "popcorn",
    "salad_dressing",
    "new_salad_dressing",
    "tomato_sauce",
    "white_bowl",
    "akita_black_bowl",
    "plate",
    "glazed_rim_porcelain_ramekin",
    "red_coffee_mug",
    "porcelain_mug",
    "white_yellow_mug",
    "chefmate_8_frypan",
    "bowl_drainer",
    "moka_pot",
    "window",
    "faucet",
    "black_book",
    "yellow_book",
    "desk_caddy",
    "wine_bottle",
]


@dataclass
class GenerateConfig:
    # pi0.5 policy server
    host: str = "127.0.0.1"
    port: int = 8001
    # Replan interval, in env steps (matches the official pi0.5 LIBERO eval client).
    replan_steps: int = 5
    # Image size expected by the pi0.5 policy (224 for pi05_libero).
    resize_size: int = 224

    # RoboCerebra environment-specific parameters
    robocerebra_root: Optional[str] = field(
        default_factory=lambda: _read_env(BENCH_ROOT_ENV))
    init_files_root: Optional[str] = field(
        default_factory=_default_init_files_root)
    task_suite_name: str = TaskSuite.ROBOCEREBRA.value
    task_types: Optional[List[str]] = None
    # Comma-separated case names to evaluate (subset filter; empty = all cases).
    cases: str = ""
    num_steps_wait: int = 15
    # Number of rollouts per task (paper protocol uses 10; their default is 5).
    num_trials_per_task: int = 5
    env_img_res: int = 256
    switch_steps: int = 150
    resume: bool = False
    dynamic_shift_description: bool = False
    complete_description: bool = False
    task_description_suffix: str = ""
    dynamic: bool = False
    use_init_files: bool = True
    initial_states_path: str = "DEFAULT"

    # Logging and utilities
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = 7

    def __post_init__(self) -> None:
        if self.task_types is None:
            self.task_types = list(DEFAULT_TASK_TYPES)

        if self.use_init_files and not self.init_files_root:
            if self.robocerebra_root:
                self.init_files_root = str(
                    Path(self.robocerebra_root) / "init_files")

        self._configure_dynamic_parameters()

    def _configure_dynamic_parameters(self) -> None:
        """Set baseline dynamic settings before per-task overrides."""
        self.dynamic = False
        self.dynamic_shift_description = False


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.robocerebra_root, ("Set --robocerebra_root or export "
                                  f"{BENCH_ROOT_ENV}.")
    if cfg.use_init_files:
        assert cfg.init_files_root, ("Set --init_files_root or export "
                                     f"{INIT_FILES_ROOT_ENV}.")
    if cfg.dynamic:
        assert cfg.resume, "dynamic=True requires resume=True."
