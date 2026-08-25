"""Request-local DiT denoising-step context used by Omega-QVLA W4A4.

The context is deliberately a no-op for the existing DuQuant path.  GPTQ
linears use it to select the activation-scale row calibrated for the current
denoising step.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional


_step_var: ContextVar[Optional[int]] = ContextVar("gr00t_dit_step", default=None)
_total_var: ContextVar[Optional[int]] = ContextVar(
    "gr00t_dit_total_steps", default=None
)


def get_current_dit_step() -> Optional[int]:
    return _step_var.get()


def get_total_dit_steps() -> Optional[int]:
    return _total_var.get()


@contextmanager
def set_dit_quant_step(
    step: Optional[int], total: Optional[int] = None
) -> Iterator[None]:
    step_token = _step_var.set(step)
    total_token = _total_var.set(total) if total is not None else None
    try:
        yield
    finally:
        _step_var.reset(step_token)
        if total_token is not None:
            _total_var.reset(total_token)
