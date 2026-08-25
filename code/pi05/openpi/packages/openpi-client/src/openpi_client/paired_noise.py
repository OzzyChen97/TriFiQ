"""GR00T-aligned deterministic diffusion noise for paired evaluation.

This deliberately mirrors ``scripts/run_robocasa365_gr00t_eval.py`` and the
server-side generator in ``scripts/inference_service.py``.  Keeping the seed
derivation and CPU Torch RNG identical is what makes a π0.5/GR00T comparison
paired at ``(task, env_seed, replan_index)`` rather than merely reproducible
within each implementation.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch


PROTOCOL = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"
DOMAIN = b"quantvla-robocasa365-v1\0"
DEFAULT_SHAPE = (50, 32)


def action_noise_seed(task: str, env_seed: int, replan_index: int) -> int:
    """Return the exact signed-63-bit request seed used by GR00T N1.5."""
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    if int(env_seed) < 0 or int(replan_index) < 0:
        raise ValueError("env_seed and replan_index must be non-negative")
    payload = (
        DOMAIN
        + task.encode("utf-8")
        + b"\0"
        + str(int(env_seed)).encode("ascii")
        + b"\0"
        + str(int(replan_index)).encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def paired_action_noise(
    task: str,
    env_seed: int,
    replan_index: int,
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
) -> np.ndarray:
    """Generate the GR00T request-local CPU ``torch.randn`` stream.

    The model-specific shape is allowed to differ (π0.5 natively predicts
    ``50x32`` while GR00T predicts its own horizon/action dimensions), but the
    seed and standard-normal generator are exactly the same protocol.
    """
    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError(f"invalid action-noise shape: {shape!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(action_noise_seed(task, env_seed, replan_index))
    value = torch.randn(
        tuple(int(dimension) for dimension in shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return value.numpy()


def paired_noise_selftest() -> None:
    first = paired_action_noise("OpenCabinet", 7, 3)
    second = paired_action_noise("OpenCabinet", 7, 3)
    assert first.shape == DEFAULT_SHAPE
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(action_noise_seed("OpenCabinet", 7, 3))
    expected = torch.randn(DEFAULT_SHAPE, generator=generator, dtype=torch.float32).numpy()
    assert np.array_equal(first, expected)
    assert not np.array_equal(first, paired_action_noise("OpenCabinet", 7, 4))
    assert not np.array_equal(first, paired_action_noise("OpenCabinet", 8, 3))


if __name__ == "__main__":
    paired_noise_selftest()
    print(f"[openpi_client.paired_noise] selftest OK ({PROTOCOL})")
