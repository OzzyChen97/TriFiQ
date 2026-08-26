#!/usr/bin/env python3
"""Shared CKA/CS formulas with a small hook-output compatibility facade."""

from __future__ import annotations

from typing import Any

import torch

from gr00t.quantization.kernel_scores import (  # single formula authority
    LayerScoreBank as _CanonicalBank,
    split_blocks,
)


def extract_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return next((value for value in output if isinstance(value, torch.Tensor)), None)
    if isinstance(output, dict):
        return next((value for value in output.values() if isinstance(value, torch.Tensor)), None)
    return None


class LayerScoreBank:
    """OpenPI hook facade over the same GR00T-authoritative CKA/CS bank."""

    def __init__(self, name: str, max_tokens: int = 256):
        self.name = name
        self.max_tokens = max_tokens
        self._bank = _CanonicalBank(name, max_tokens=max_tokens)

    def accumulate_reference(self, output: Any) -> None:
        value = extract_tensor(output)
        if value is None:
            return
        front, back = split_blocks(value, self.max_tokens)
        zero_mask = back.norm(dim=1) > 1e-9 if back is not None else None
        self._bank.accumulate_ref_blocks(front, back, zero_mask)

    def finalize(self) -> None:
        self._bank.finalize_ref()

    def evaluate(self, outputs: list[Any]) -> dict[str, float | None]:
        chunks = []
        for output in outputs:
            value = extract_tensor(output)
            if value is not None:
                chunks.append(split_blocks(value, self.max_tokens))
        fronts = [front for front, _ in chunks if front is not None]
        backs = [back for _, back in chunks if back is not None]
        front = torch.cat(fronts, dim=0) if fronts else None
        back = torch.cat(backs, dim=0) if backs else None
        return self._bank.evaluate_blocks(front, back)

    def evaluate_aligned_rows(self, rows: torch.Tensor) -> dict[str, float | None]:
        return self._bank._evaluate_rows(rows.detach().to(dtype=torch.float32, device="cpu"))

    @property
    def ready(self) -> bool:
        return self._bank.ready

    @property
    def _reference(self) -> torch.Tensor | None:
        return self._bank._fp_raw

    def metadata(self) -> dict[str, Any]:
        return self._bank.state_dict()


def selftest() -> None:
    value = torch.randn(4, 32, 64, generator=torch.Generator().manual_seed(0))
    bank = LayerScoreBank("shared", max_tokens=128)
    bank.accumulate_reference(value)
    bank.finalize()
    same = bank.evaluate([value])
    assert same["cka"] is not None and abs(same["cka"] - 1.0) < 1e-4
    assert same["cs"] is not None and same["cs"] < 1e-4
    print("[quantvla-kernel-scores] selftest OK")


if __name__ == "__main__":
    selftest()
