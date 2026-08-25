# π0.5 RoboCasa365 Table 1

Matrix complete: **True**. Expected episodes/config: 2500.

| Config | Complete | Successes | Episode SR | Held-out 46 macro SR | All 50 macro SR |
|---|---:|---:|---:|---:|---:|
| fp16 | 2500/2500 | 654 | 0.262 | 0.229 | 0.262 |
| quantvla_w4a8_atmohb | 2500/2500 | 621 | 0.248 | 0.219 | 0.248 |
| gdsq_vla_atmohb | 2500/2500 | 664 | 0.266 | 0.243 | 0.266 |
| gdsq_vla | 2500/2500 | 633 | 0.253 | 0.230 | 0.253 |

## Task-set macro SR

| Task set | fp16 | quantvla_w4a8_atmohb | gdsq_vla_atmohb | gdsq_vla |
|---|---:|---:|---:|---:|
| atomic_seen | 0.583 | 0.561 | 0.571 | 0.549 |
| composite_seen | 0.140 | 0.128 | 0.150 | 0.150 |
| composite_unseen | 0.021 | 0.018 | 0.038 | 0.024 |

## Prespecified paired comparisons

| Comparison | Delta | 95% CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|
| gdsq_vla_atmohb_vs_gdsq_vla | +0.013 | [-0.004, +0.031] | 0.1808 | 0.5424 |
| gdsq_vla_vs_fp16 | +0.001 | [-0.016, +0.018] | 0.96 | 0.96 |
| gdsq_vla_vs_quantvla_w4a8_atmohb | +0.011 | [-0.009, +0.030] | 0.2895 | 0.579 |
| gdsq_vla_atmohb_vs_quantvla_w4a8_atmohb | +0.024 | [+0.005, +0.044] | 0.0232 | 0.0928 |

## Efficiency

| Config | Episode wall (s) | Inference/replan (s) | Server infer (ms) | Peak server CUDA MiB | Mean GPU util |
|---|---:|---:|---:|---:|---:|
| fp16 | 128.0 | 0.227 | 196.5 | nan | nan% |
| quantvla_w4a8_atmohb | 144.4 | 0.376 | 327.9 | nan | nan% |
| gdsq_vla_atmohb | 136.2 | 0.307 | 264.9 | nan | nan% |
| gdsq_vla | 134.9 | 0.294 | 255.3 | 20108 | 24.0% |

## Paper-style candidate-component memory

| Config | GiB | Compression vs FP16 |
|---|---:|---:|
| fp16 | 4.113 | 1.00× |
| quantvla_w4a8_atmohb | 1.388 | 2.96× |
| gdsq_vla_atmohb | 1.868 | 2.20× |
| gdsq_vla | 1.868 | 2.20× |

GDSQ quantizes 1,788,870,656/2,208,301,056 candidate parameters (81.0%) across 80 W4 + 100 FP16 layers.

Paper-style memory is tightly packed theoretical component storage; eager fake-quant CUDA residency is reported separately and is not presented as deployment compression.

LIBERO context only: v1.4 89.2%, uniform W6 88.2%, v1.3 85.2%; not pooled with RoboCasa365.
