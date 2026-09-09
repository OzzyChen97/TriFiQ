# DyPAC-VLA — minimal anonymous artifact

Anonymous artifact for the ICLR 2027 submission **DyPAC-VLA: Representation-to-Action Fidelity
for Low-Bit Vision-Language-Action Models**.

This release is deliberately minimal but runnable. It contains the frozen method core and the paper
source, so that a reader can (i) execute the self-tests of every shipped component, (ii) run a
self-contained W4A8 demo that exercises the quantizer and the D-PAC metric, and (iii) rebuild the
paper PDF. Large calibration buffers, model checkpoints, simulator stacks, and the full evaluation
harness are intentionally not redistributed.

## Contents

```
scripts/quantvla_cross_model_protocol.json     frozen cross-model protocol (single source of truth)
scripts/quantvla_dynamic_a8_protocol.json      frozen DyRange-A8 runtime contract
scripts/quantvla_full_context_protocol.json    frozen FCP protocol, v1
scripts/quantvla_full_context_protocol_v2.json frozen FCP protocol, v2
scripts/tools/quantvla_metric_protocol.py      D-PAC / D_func action-chunk metric
scripts/tools/gr00t_func_metrics.py            SE(3) action metrics used by the D-PAC core
scripts/tools/quantvla_hessian_w4.py           Hessian-aware group-64 W4 + A8 scale tables
scripts/tools/quantvla_dynamic_a8_protocol.py  DyRange-A8 runtime contract validation
scripts/tools/quantvla_full_context.py         full-context precision protection (FCP) selection
scripts/tools/quantvla_table1_bytes.py         exact static byte accounting and budget rules
scripts/tools/quantvla_outputimpact.py         output-impact scoring utilities
scripts/tools/aggregate_full_context_{quick,table1}.py  frozen FCP statistics (hash-pinned)
examples/run_selftests.py                      runs every self-test that ships with the release
examples/demo_w4a8.py                          self-contained W4A8 + D-PAC demo
paper/                                         LaTeX source and compiled PDF of the submission
```

## Quick start

```bash
pip install -r requirements.txt

make selftest   # run the shipped self-tests (CPU only)
make demo       # Hessian W4 + dynamic A8 + D-PAC on synthetic data (CPU only)
make paper      # rebuild paper/main.pdf (requires pdflatex + bibtex)
```

`make demo` needs no model, dataset, or GPU. It quantizes a synthetic `Linear` layer with the frozen
group-64 Hessian W4 routine, builds a deterministic per-flow-step A8 scale table, compares the W4A8
matmul against FP16, reports the D-PAC/D_func of a perturbed action chunk, and prints the paper's
static byte accounting.

## Method in one paragraph

DyPAC-VLA is a training-free mixed-precision post-training quantization framework for
vision-language-action policies. It (1) measures **prefix-accumulated control divergence (D-PAC)**,
a closed-loop action-chunk metric over local error, prefix accumulation, composed SE(3) drift,
inter-replan discontinuity, gripper timing, and tail risk; (2) selects an exact-budget W4/FP16 mask
by **full-context precision protection (FCP)**, which accepts a change only after whole-network,
cross-context adjudication with component-safety checks; and (3) deploys the frozen mask with
**Hessian-aware group-64 W4 weights** and **DyRange-A8** deterministic per-forward activation
scales. No task routing, success-label feedback, retraining, runtime selector, or action correction
is used.

## Scope and omissions

Included: the frozen protocols, the method core listed above, and the paper.

Not included, because they are data or infrastructure rather than method:

- the frozen calibration and on-policy probe buffers (117 MB and 11 MB `npz` files) referenced by
  `scripts/quantvla_cross_model_protocol.json`. Consequently
  `python scripts/tools/quantvla_cross_model_protocol.py` (its own `selftest`) reports a missing
  artifact; every other shipped self-test runs. `make selftest` runs the components that are
  self-contained and prints this limitation explicitly;
- model-family integrations for GR00T N1.5 and pi0.5, simulator harnesses, and the sweep drivers
  that produced the paper's tables;
- model checkpoints, rollout outputs, and run directories.

The offline FCP statistics scripts are hash-pinned by `quantvla_full_context.protocol_attestation()`
and are therefore included for attestation, but they require frozen run directories to execute.

## License

Apache-2.0; see `LICENSE`. Third-party TeX compatibility file under `paper/third_party/` keeps its
upstream license (see `paper/third_party/README.md`).
