#!/usr/bin/env python3
"""Historical import tombstone for the removed π0.5-only metric path.

New calibration, sensitivity, mask selection, and formal evaluation code must
call :func:`quantvla_metric_protocol.summarize_pair` on inverse-normalized
physical actions.  Keeping this tiny module makes old artifact readers able to
resolve formula constants while failing closed if they attempt the obsolete
native-trajectory metric calls.
"""

from __future__ import annotations

from typing import NoReturn

from quantvla_metric_protocol import (
    ACTION_HORIZON as EXECUTED_ACTIONS,
    FUNCTIONAL_FORMULA_ID,
    PAC_FORMULA_ID,
    summarize_pair,
)


ACTION_HORIZON = 50
FLOW_STEPS = 4
HISTORICAL_ONLY = True


def _removed() -> NoReturn:
    raise RuntimeError(
        "pi05_func_metrics was removed: inverse-normalize final actions in the "
        "pi0.5 adapter and call quantvla_metric_protocol.summarize_pair"
    )


def d_func(*_args, **_kwargs) -> NoReturn:
    _removed()


def d_pac_sequence(*_args, **_kwargs) -> NoReturn:
    _removed()


def selftest() -> None:
    assert summarize_pair is not None
    try:
        d_func(None, None)
    except RuntimeError as error:
        assert "summarize_pair" in str(error)
    else:
        raise AssertionError("obsolete π0.5 metric path did not fail closed")
    print("[pi05_func_metrics] historical path removed; shared summarize_pair required")


if __name__ == "__main__":
    selftest()
