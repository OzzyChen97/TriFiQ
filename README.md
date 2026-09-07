# DyPAC-VLA

### Full-Context Mixed-Precision Quantization for Vision-Language-Action Models

[Apache-2.0](LICENSE)

**DyPAC-VLA** (*Dynamic-Range and Prefix-Accumulated Control-aware Quantization*) is a
training-free mixed-precision post-training quantization framework for vision-language-action
policies. It evaluates precision decisions in the complete policy, freezes one global W4/FP16 mask
under an exact byte budget, and deploys the mask with Hessian-aware group-64 W4 weights and dynamic
per-forward A8 ranges.

DyPAC-VLA uses no task routing, success-label feedback, retraining, runtime selector, or action
correction.

## Architecture

1. **Prefix-Accumulated Control Divergence (D-PAC).** Measures local action error, prefix
   accumulation, composed SE(3) pose drift, inter-replan discontinuity, gripper-event timing, and
   tail risk on paired FP16 and quantized action chunks.
2. **Full-Context Precision Protection (FCP).** Generates exact-budget W4/FP16 candidates and
   accepts changes only after whole-network, cross-context adjudication with component-safety
   checks.
3. **DyRange-A8 deployment.** Combines signed-nibble group-64 Hessian W4 weights with deterministic
   per-forward, per-input-channel A8 scales.

## RoboCasa365 Results

All reported configurations use the complete 50-task target split with 50 scenarios per task.
The pi0.5 DyPAC-VLA row uses four flow-matching integration steps per action prediction, matching
the corrected frozen protocol and result metadata.

| Policy | Atomic | Composite-Seen | Composite-Unseen | Overall | Static size | Compression |
|---|---:|---:|---:|---:|---:|---:|
| GR00T N1.5 DyPAC-VLA | **74.7%** | **43.9%** | **40.9%** | **54.0%** (1350/2500) | 0.896 GiB | 2.22x |
| GR00T N1.5 DyPAC-VLA (max rate) | 70.9% | 42.9% | 42.3% | 52.8% (1319/2500) | 0.779 GiB | 2.56x |
| pi0.5 DyPAC-VLA | **59.6%** | **15.4%** | **4.3%** | **27.7%** (693/2500) | 1.523 GiB | 2.70x |
| pi0.5 DyPAC-VLA (max rate) | 58.7% | 14.5% | 3.3% | 26.8% (670/2500) | 1.157 GiB | 3.56x |

For GR00T N1.5, DyPAC-VLA improves over QuantVLA W4A8 by 23.6 percentage points under
the paired protocol. The difference from the 55.1% FP16 teacher is not statistically significant
(`p=0.2929`). For pi0.5, the 27.7% versus 26.2% FP16 comparison is descriptive because no paired
significance claim is registered for those rows.

Compression denotes exact packed static model-component storage. A separate A40, batch-one,
model-only measurement shows that the retained GR00T plan lowers median CUDA peak allocation from
5.386 to 4.267 GiB relative to native FP16, while the generic packed-W4 backend is 2.16x slower
and consumes 2.14x more gross board energy per request. The paper appendix separately audits a
real-integer W4A8 backend under a paired hardware protocol; it lowers p50 latency by 18.4% and
gross board energy by 11.8% relative to its matched reference, with unchanged peak allocation.

The GR00T FCP decision was frozen before a preregistered 3,000-episode diagnostic of all six
rejected proposals. Their 50-task success rates span 50.6--56.2% versus 53.8% for the paired frozen
anchor; none differs after Holm correction (minimum adjusted `p=0.891`). These outcomes do not
revise the selected mask, and non-significance is not treated as equivalence.

For pi0.5, a separate same-protocol FCP diagnostic completes 1,500 new episodes and reuses the
500-episode projected anchor. The budget-feasible 121-W4/59-FP16 anchor records 28.2% versus 26.0%
for the over-budget 80-W4/100-FP16 initializer (`p=0.235`). The two later proposals record 24.8%
and 25.0% (minimum Holm-adjusted `p=0.086`) and remain rejected. These diagnostic outcomes cannot
revise the frozen formal row.

A formal-scale maximal-rate row is displayed in Table 1 as **Ours (max rate)** for each
model: the GR00T all-W4 endpoint (116 W4, 0.779 GiB, 2.56x) records **52.8%** (1319/2500), and
the pi0.5 180-layer all-W4 profile (1.157 GiB, 3.56x) records **26.8%** (670/2500), under the
same four-flow-step protocol. Cross-run paired against the audited anchors (54.0% GR00T; 27.7%
pi0.5), the discordant counts are 319/288 (exact McNemar `p=0.223`) and 182/159 (`p=0.233`).
Both are descriptive operating points---not registered tests---and neither revises the frozen
headline rows. The combined aggregate is `runs/robocasa365_table1_max_sweep_v1/aggregate.json`
(sha256 `93bb7b021ca83b073b314c6b18e97f3d7b4c4e327eeb190375d7d5fedb3070bc`); the plans are
`runs/robocasa365_table1_max_sweep_v1/masks/gr00t_max.plan.json` (sha256
`b1fe584404184bf6f5505af5ef40ebaeab759a94c1e0e952dd19b085e387179d`) and
`runs/robocasa365_table1_max_sweep_v1/masks/pi05_max.plan.json` (sha256
`8b0aabc15e48970250e605ca689e08bde3c9b73616d2c31fd2fae88d01759dbd`).

## Repository Layout

- `scripts/`: evaluation, calibration, scheduling, and inference entry points.
- `scripts/tools/`: plan construction, aggregation, auditing, and quantization utilities.
- `code/`: model-family integrations and evaluation backends.
- `environments/`: environment specifications.
- `tests/`: unit and protocol tests.
- `assets/`: non-paper project assets.
- `docs/gdsq_vla_iclr2027/`: paper source, generated figures and tables, and the compiled PDF.

Large model checkpoints, datasets, rollout outputs, and export caches are intentionally excluded
from version control.

## Installation

Create the model-specific environments described under `environments/`, then install the project
in editable mode:

```bash
pip install -e .
```

Run the repository-level checks with:

```bash
make run-checks
```

GR00T and pi0.5 use separate runtime environments; evaluation scripts under `scripts/` activate
the appropriate backend and preserve frozen protocol metadata.

## Reproducibility and Scope

Frozen manifests bind model plans, task/seed coverage, environment settings, and artifact hashes.
Formal result cells require complete coverage with no missing, duplicate, conflicting, or failed
episode rows.

The eager fake-quant path validates numerical behavior and exact static storage accounting. The
separate A40 measurements show a model-only peak-allocation reduction for the generic backend and
a latency/energy improvement for a separately audited real-integer backend. The latter changes A8
scale granularity and therefore remains an appendix-only deployment point rather than replacing the
formal Ours row. Other accelerators, larger batches, end-to-end or robot energy, and physical-robot
transfer remain outside the present claim scope.

The paper source and compiled PDF are versioned under `docs/gdsq_vla_iclr2027/`. Run `make check`
in that directory to regenerate the paper and validate its registered evidence.
