#!/usr/bin/env python3
"""Adapter from the LIBERO multi-GPU harness to the current pi0.5 runtime."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys

import tyro


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE = REPO_ROOT / "code/pi05/openpi/scripts/serve_pi05_quant_policy.py"


@dataclasses.dataclass
class Args:
    model_path: str
    data_config: str = "pi05_libero"
    port: int = 8000
    denoising_steps: int = 8
    server: bool = False
    embodiment_tag: str = "new_embodiment"
    device: str = "cuda"


def load_service():
    spec = importlib.util.spec_from_file_location("quantvla_table6_pi05_service", SERVICE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pi0.5 service: {SERVICE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(args: Args) -> None:
    if args.device != "cuda":
        raise ValueError("Table 6 pi0.5 service requires device=cuda")
    service = load_service()
    service.main(
        service.Args(
            env=service.EnvMode.LIBERO,
            port=args.port,
            denoising_steps=args.denoising_steps,
            policy=service.Checkpoint(config=args.data_config, dir=args.model_path),
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
