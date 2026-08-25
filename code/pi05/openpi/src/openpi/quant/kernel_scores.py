"""Paired block-level CKA/CS and amplitude guards for the π0.5 port."""

from __future__ import annotations

import math
from typing import Optional

import torch


def _subsample(rows: torch.Tensor, cap: int) -> torch.Tensor:
    if rows.shape[0] <= cap:
        return rows.contiguous()
    indices = torch.linspace(
        0, rows.shape[0] - 1, cap, device=rows.device
    ).round().long()
    return rows.index_select(0, indices).contiguous()


def split_blocks(
    tensor: torch.Tensor,
    max_tokens: int,
    front_fraction: float = 0.5,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Position-stratified, deterministic row selection on CPU float32."""
    if tensor is None:
        return None, None
    value = extract_tensor(tensor)
    if value is None:
        return None, None
    value = value.detach().to(dtype=torch.float32)
    if value.ndim < 3:
        rows = value.reshape(-1, value.shape[-1])
        return (_subsample(rows, max_tokens).cpu(), None) if rows.shape[0] >= 2 else (None, None)
    value = value.reshape(-1, value.shape[-2], value.shape[-1])
    length = value.shape[1]
    split = min(length, max(1, round(length * front_fraction)))
    cap = max(2, max_tokens // 2)
    front = _subsample(value[:, :split].reshape(-1, value.shape[-1]), cap).cpu()
    back_rows = value[:, split:].reshape(-1, value.shape[-1])
    back = _subsample(back_rows, cap).cpu() if back_rows.shape[0] else None
    return front, back


def extract_tensor(output) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            found = extract_tensor(value)
            if found is not None:
                return found
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return None


def _center(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=0, keepdim=True)


def _log_mean_gaussian(
    left: torch.Tensor,
    right: Optional[torch.Tensor],
    sigma: float,
    block: int = 128,
) -> float:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x = left.to(device)
    y = x if right is None else right.to(device)
    total = None
    denominator = 2.0 * sigma * sigma
    for start in range(0, x.shape[0], block):
        distances = torch.cdist(x[start : start + block], y).square()
        partial = (-distances / denominator).logsumexp(dim=(0, 1))
        total = partial if total is None else torch.logaddexp(total, partial)
    if total is None:
        raise ValueError("empty kernel input")
    return float(total) - math.log(x.shape[0] * y.shape[0])


class LayerScoreBank:
    """Reference cache for row-paired block-level linear CKA and CS divergence."""

    def __init__(self, name: str, max_tokens: int = 256):
        self.name = name
        self.max_tokens = max_tokens
        self._front: list[torch.Tensor] = []
        self._back: list[torch.Tensor] = []
        self._back_masks: list[torch.Tensor] = []
        self._row_keep: Optional[torch.Tensor] = None
        self._reference: Optional[torch.Tensor] = None
        self._centered: Optional[torch.Tensor] = None
        self._reference_gram: Optional[torch.Tensor] = None
        self._self_norm: Optional[float] = None
        self._log_self: Optional[float] = None
        self.sigma: Optional[float] = None

    def accumulate_reference(self, output) -> None:
        front, back = split_blocks(output, self.max_tokens)
        if front is not None:
            self._front.append(front)
        if back is not None:
            self._back.append(back)
            self._back_masks.append(back.norm(dim=1) > 1e-9)

    def finalize(self) -> None:
        if not self._front:
            return
        front = torch.cat(self._front, dim=0)
        back = torch.cat(self._back, dim=0) if self._back else None
        rows = torch.cat([front, back], dim=0) if back is not None else front
        keep = torch.ones(rows.shape[0], dtype=torch.bool)
        if back is not None and self._back_masks:
            keep[front.shape[0] :] = torch.cat(self._back_masks, dim=0)
        kept_indices = torch.where(keep)[0]
        if kept_indices.numel() > self.max_tokens:
            selection = torch.linspace(0, kept_indices.numel() - 1, self.max_tokens).round().long()
            kept_indices = kept_indices.index_select(0, selection)
        row_keep = torch.zeros(rows.shape[0], dtype=torch.bool)
        row_keep[kept_indices] = True
        self._row_keep = row_keep
        reference = rows[row_keep].contiguous()
        if reference.shape[0] < 2:
            return
        self._reference = reference
        centered = _center(reference)
        self._centered = centered
        # Sample-Gram form is algebraically identical to feature-covariance
        # linear CKA and avoids materializing 2048x2048 matrices when N<=256.
        reference_gram = centered @ centered.T
        self._reference_gram = reference_gram
        self._self_norm = float(torch.linalg.matrix_norm(reference_gram, ord="fro"))
        self.sigma = self._estimate_sigma(reference)
        self._log_self = _log_mean_gaussian(reference, None, self.sigma)
        self._front.clear()
        self._back.clear()
        self._back_masks.clear()

    @staticmethod
    def _estimate_sigma(rows: torch.Tensor, pairs: int = 2048) -> float:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        value = rows.to(device)
        generator = torch.Generator(device=device).manual_seed(0)
        left = torch.randint(0, value.shape[0], (pairs,), generator=generator, device=device)
        right = torch.randint(0, value.shape[0], (pairs,), generator=generator, device=device)
        right = torch.where(left == right, (right + 1) % value.shape[0], right)
        return max(float((value[left] - value[right]).norm(dim=1).median()), 1e-3)

    def evaluate(self, outputs: list) -> dict[str, Optional[float]]:
        result = {"cka": None, "cs": None, "cs_cross": None}
        if not self.ready:
            return result
        fronts: list[torch.Tensor] = []
        backs: list[torch.Tensor] = []
        for output in outputs:
            front, back = split_blocks(output, self.max_tokens)
            if front is not None:
                fronts.append(front)
            if back is not None:
                backs.append(back)
        if not fronts:
            return result
        front = torch.cat(fronts, dim=0)
        back = torch.cat(backs, dim=0) if backs else None
        rows = torch.cat([front, back], dim=0) if back is not None else front
        if self._row_keep is None or rows.shape[0] != self._row_keep.shape[0]:
            raise RuntimeError(
                f"{self.name}: reference/intervention row mismatch "
                f"({None if self._row_keep is None else self._row_keep.shape[0]} vs {rows.shape[0]})"
            )
        rows = rows[self._row_keep].contiguous()
        return self.evaluate_aligned_rows(rows)

    def evaluate_aligned_rows(self, rows: torch.Tensor) -> dict[str, Optional[float]]:
        """Score rows already aligned to the finalized reference selection.

        This is used by the final protocol's CS in-situ scale-response check;
        normal intervention scoring should continue to call :meth:`evaluate`.
        """
        result = {"cka": None, "cs": None, "cs_cross": None}
        if not self.ready:
            return result
        rows = rows.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if self._reference is None or rows.shape != self._reference.shape:
            raise RuntimeError(
                f"{self.name}: aligned row shape mismatch "
                f"({tuple(rows.shape)} vs "
                f"{None if self._reference is None else tuple(self._reference.shape)})"
            )
        centered = _center(rows)
        quant_gram = centered @ centered.T
        cross_norm_sq = float((self._reference_gram * quant_gram).sum())
        quant_norm = float(torch.linalg.matrix_norm(quant_gram, ord="fro"))
        denominator = self._self_norm * quant_norm
        if denominator > 1e-12:
            result["cka"] = min(max(cross_norm_sq / denominator, 0.0), 1.0)
        log_yy = _log_mean_gaussian(rows, None, self.sigma)
        log_xy = _log_mean_gaussian(self._reference, rows, self.sigma)
        result["cs_cross"] = -2.0 * log_xy
        result["cs"] = max(self._log_self + log_yy - 2.0 * log_xy, 0.0)
        return result

    @property
    def ready(self) -> bool:
        return self._reference is not None and self.sigma is not None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "rows": 0 if self._reference is None else int(self._reference.shape[0]),
            "features": 0 if self._reference is None else int(self._reference.shape[1]),
            "sigma": self.sigma,
        }


def reference_guard_stats(outputs: list, max_tokens: int = 1024) -> dict:
    rows = []
    for output in outputs:
        tensor = extract_tensor(output)
        if tensor is not None:
            rows.append(
                _subsample(
                    tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1, tensor.shape[-1]),
                    max_tokens,
                )
            )
    if not rows:
        return {}
    value = _subsample(torch.cat(rows, dim=0), max_tokens)
    return {
        "rms": value.square().mean(dim=0).sqrt(),
        "amax": float(value.abs().max()),
        "p999": float(torch.quantile(value.abs().flatten(), 0.999)),
    }


def guard_metrics(reference: dict, outputs: list, max_tokens: int = 1024) -> dict:
    quant = reference_guard_stats(outputs, max_tokens=max_tokens)
    if not reference or not quant:
        return {"rms_ratio": None, "amax_ratio": None, "sat_rate": None}
    ref_rms = reference["rms"]
    quant_rms = quant["rms"]
    rms_ratio = ((quant_rms + 1e-6) / (ref_rms + 1e-6)).log().abs().median()
    values = []
    for output in outputs:
        tensor = extract_tensor(output)
        if tensor is not None:
            values.append(tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1))
    flat = _subsample(torch.cat(values)[:, None], max_tokens * 64).flatten()
    return {
        "rms_ratio": float(rms_ratio),
        "amax_ratio": float(quant["amax"] / max(reference["amax"], 1e-6)),
        "sat_rate": float((flat.abs() > reference["p999"]).to(torch.float32).mean()),
    }


def selftest() -> None:
    torch.manual_seed(0)
    value = torch.randn(4, 32, 64)
    bank = LayerScoreBank("test", max_tokens=128)
    bank.accumulate_reference(value)
    bank.finalize()
    same = bank.evaluate([value])
    scaled = bank.evaluate([value * 4])
    assert same["cka"] is not None and abs(same["cka"] - 1.0) < 1e-4
    assert same["cs"] is not None and same["cs"] < 1e-4
    assert scaled["cka"] is not None and abs(scaled["cka"] - 1.0) < 1e-3
    assert scaled["cs"] is not None and scaled["cs"] > same["cs"]
    reference = reference_guard_stats([value])
    guards = guard_metrics(reference, [value * 4])
    assert abs(guards["rms_ratio"] - math.log(4)) < 1e-3
    assert guards["sat_rate"] > 0
    print("[openpi.quant.kernel_scores] selftest OK")


if __name__ == "__main__":
    selftest()
