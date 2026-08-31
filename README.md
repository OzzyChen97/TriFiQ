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
| pi0.5 DyPAC-VLA | **59.6%** | **15.4%** | **4.3%** | **27.7%** (693/2500) | 1.523 GiB | 2.70x |

For GR00T N1.5, DyPAC-VLA improves over QuantVLA W4A8 by 23.6 percentage points under
the paired protocol. The difference from the 55.1% FP16 teacher is not statistically significant
(`p=0.2929`). For pi0.5, the 27.7% versus 26.2% FP16 comparison is descriptive because no paired
significance claim is registered for those rows.

Compression denotes exact packed static model-component storage, not end-to-end latency or peak
live memory.

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

The current eager fake-quant implementation validates numerical behavior and exact static storage
accounting. Fused low-bit kernel latency, peak-memory improvements, and real-robot transfer remain
outside the present claim scope.

The paper source and compiled PDF are versioned under `docs/gdsq_vla_iclr2027/`. Run `make check`
in that directory to regenerate the paper and validate its registered evidence.
