# DyPAC-VLA

### Full-Context Mixed-Precision Quantization for Vision-Language-Action Models

[Under review — ICLR 2027 anonymous submission] · [Apache-2.0](LICENSE)

**DyPAC-VLA** (*Dynamic-Range and Prefix-Accumulated Control-aware Quantization*) is a
training-free mixed-precision PTQ framework for VLA policies. It evaluates precision choices in
the complete policy, freezes one global W4/FP16 mask under an exact byte budget, and deploys the
mask with Hessian-aware group-64 W4 weights and dynamic per-forward A8 ranges. It uses no task
routing, success labels, retraining, or runtime correction.

The paper source remains in the legacy directory
[`docs/gdsq_vla_iclr2027`](docs/gdsq_vla_iclr2027/) so existing build and Overleaf automation do
not break; the manuscript identity and all active labels are **DyPAC-VLA**.

## Architecture

1. **Prefix-Accumulated Control Divergence ($D_{PAC}$).** Compares paired FP16 and quantized
   action chunks using local error, prefix accumulation, composed SE(3) pose drift, inter-replan
   stitching, gripper-event timing, and CVaR tail risk.
2. **Full-Context Precision Protection (FCP).** Generates exact-budget counterfactual W4/FP16
   plans, then lets whole-network cross-context rescoring and component safety decide.
3. **Dynamic-range deployment.** Uses signed-nibble group-64 Hessian W4 and **DyRange-A8**, which
   recomputes each input-channel scale on every forward call.

For GR00T, FCP retains the initialized 100-W4/16-FP16 mask as a local optimum in the tested
one-layer-flip neighborhood. The audit finds no positive-benefit single-layer flip and no eligible
structured alternative; this is not claimed as a newly discovered mask or a global optimum.

## Frozen Results

| Model / role | W4 / FP16 | Success | Static bytes | Compression | Claim |
|---|---:|---:|---:|---:|---|
| GR00T N1.5 / DyPAC-VLA (ours) | 100 / 16 | **54.0%** (1350/2500) | 962,068,480 | 2.224× | +23.6pp over QuantVLA, Holm $p<10^{-4}$; difference from FP16 is not significant |
| $\pi_{0.5}$ / compression anchor | 121 / 59 | 29/100 screen | 1,634,828,288 | 2.702× | Compression only; no success-rate superiority claim |

The GR00T split success rates are 74.7% Atomic-Seen, 43.9% Composite-Seen, and 40.9%
Composite-Unseen. FP16 obtains 55.1%; the paired comparison with ours has $p=0.2929$ and a
95% hierarchical-bootstrap interval of [-3.88, 1.76] points.

The paper places audited core mechanism ablations in the main text. The pending LIBERO comparison
and lower-priority planned diagnostic ablations are appendix-only and explicitly claim-disabled.

## Paper and Evidence

- Paper: [`docs/gdsq_vla_iclr2027/main.tex`](docs/gdsq_vla_iclr2027/main.tex)
- Frozen version map: [`FINAL_VERSIONS.md`](FINAL_VERSIONS.md)
- Generated evidence registry:
  [`docs/gdsq_vla_iclr2027/dypac_evidence_registry.json`](docs/gdsq_vla_iclr2027/dypac_evidence_registry.json)
- GR00T frozen plan: `runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json`
- GR00T formal aggregate: `runs/full_context_v2/table1/aggregate.json`
- $\pi_{0.5}$ frozen plan: `runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json`

Build and audit the anonymous paper with:

```bash
make test-paper
```

Regenerate or verify paper cells with:

```bash
/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python \
  scripts/tools/render_dypac_paper.py
/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python \
  scripts/tools/render_dypac_paper.py --check
```

## Scope

Reported compression is exact packed **static model-component** storage. The eager fake-quant
implementation does not establish end-to-end latency or live-memory gains. The $\pi_{0.5}$ result
is deliberately kept outside the formal headline because its quick screen did not pass the
success-rate gate.
