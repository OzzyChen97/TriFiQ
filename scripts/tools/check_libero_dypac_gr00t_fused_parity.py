#!/usr/bin/env python3
"""Check deployed GR00T packed W4 against explicit dynamic-A8 dequantized math."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gr00t.quantization.duquant_fused import fused_linear_w4_nibbles
from quantvla_hessian_w4 import unpack_signed_nibbles
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    hessian = Path(args.hessian).resolve()
    metadata = json.loads(Path(str(hessian) + ".json").read_text(encoding="utf-8"))
    if metadata.get("protocol_id") != PROTOCOL["protocol_id"]:
        raise ValueError("GR00T fused parity Hessian protocol drift")
    rows = []
    generator = torch.Generator().manual_seed(20260831)
    with np.load(hessian, allow_pickle=False) as archive:
        names = [str(value) for value in archive["layer_names"].tolist()]
        indices = sorted(
            set(
                np.linspace(0, len(names) - 1, min(12, len(names)))
                .round()
                .astype(int)
                .tolist()
            )
        )
        for index in indices:
            packed = torch.from_numpy(np.asarray(archive[f"packed_{index:04d}"])).to(args.device)
            scales = torch.from_numpy(np.asarray(archive[f"scales_{index:04d}"])).to(args.device)
            in_features = packed.shape[1] * 2
            codes = unpack_signed_nibbles(packed.cpu(), in_features).to(
                args.device, torch.float32
            )
            dequant = codes * scales.float().repeat_interleave(64, dim=1)[:, :in_features]
            for count in (1, 17, 65):
                x = torch.randn(
                    count, in_features, generator=generator, dtype=torch.float16
                ).to(args.device)
                activation_scale = (
                    x.float().abs().amax(dim=0) / 127.0
                ).clamp_min(1e-6).to(torch.float16)
                x_quant = (
                    torch.clamp(torch.round(x / activation_scale), -128, 127)
                    * activation_scale
                )
                expected = (x_quant.float() @ dequant.T).to(torch.float16)
                actual = fused_linear_w4_nibbles(x_quant, packed, scales)
                delta = (actual.float() - expected.float()).abs()
                row = {
                    "layer": names[index],
                    "rows": count,
                    "max_abs": float(delta.max()),
                    "mean_abs": float(delta.mean()),
                    "all_finite": bool(torch.isfinite(actual).all()),
                }
                if not row["all_finite"] or not torch.allclose(
                    actual, expected, atol=2e-2, rtol=2e-2
                ):
                    raise RuntimeError(f"GR00T fused parity failed: {row}")
                rows.append(row)
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_gr00t_dynamic_a8_fused_parity",
        "model": "gr00t",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "hessian": str(hessian),
        "hessian_sha256": sha256_file(hessian),
        "tested_layers": len(set(row["layer"] for row in rows)),
        "tested_shapes": len(rows),
        "maximum_absolute_error": max(row["max_abs"] for row in rows),
        "passed": True,
        "rows": rows,
    }
    atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "passed",
                    "tested_layers",
                    "tested_shapes",
                    "maximum_absolute_error",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
