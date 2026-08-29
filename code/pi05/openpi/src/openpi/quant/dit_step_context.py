"""Request-local deterministic flow-step index for pi0.5 DiT A8 tables."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional


_step: ContextVar[Optional[int]] = ContextVar("openpi_dit_step", default=None)
_total: ContextVar[Optional[int]] = ContextVar("openpi_dit_total_steps", default=None)


def get_current_dit_step() -> Optional[int]:
    return _step.get()


def get_total_dit_steps() -> Optional[int]:
    return _total.get()


@contextmanager
def set_dit_quant_step(step: Optional[int], total: Optional[int] = None) -> Iterator[None]:
    step_token = _step.set(step)
    total_token = _total.set(total) if total is not None else None
    try:
        yield
    finally:
        _step.reset(step_token)
        if total_token is not None:
            _total.reset(total_token)
