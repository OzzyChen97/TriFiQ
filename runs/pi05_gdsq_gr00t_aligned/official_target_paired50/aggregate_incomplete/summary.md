# π0.5 RoboCasa365 Table 1

Matrix complete: **False**. Expected episodes/config: 2500.

| Config | Complete | Successes | Episode SR | Held-out 46 macro SR | All 50 macro SR |
|---|---:|---:|---:|---:|---:|
| fp16 | 704/2500 | 437 | 0.621 | pending | pending |
| quantvla_w4a8_atmohb | 615/2500 | 356 | 0.579 | pending | pending |
| gdsq_vla_atmohb | 647/2500 | 384 | 0.594 | pending | pending |
| gdsq_vla | 1922/2500 | 618 | 0.322 | pending | pending |

## Progress diagnostics (incomplete matrix, not comparable results)

Incomplete matrix: configs advance through task sets at different rates, so pooled Episode SR mixes unequal task difficulty and must never be compared across configs. Use the stratified and paired-common-key views for progress monitoring only; neither is a formal result.

| Config | atomic_seen | composite_seen | composite_unseen |
|---|---:|---:|---:|
| fp16 | 437/704 = 0.621 | - | - |
| quantvla_w4a8_atmohb | 356/615 = 0.579 | - | - |
| gdsq_vla_atmohb | 384/647 = 0.594 | - | - |
| gdsq_vla | 494/900 = 0.549 | 120/800 = 0.150 | 4/222 = 0.018 |

Paired SR on the 615 (task, seed) keys completed by all four configs (atomic_seen=615):

| Config | Paired common-key SR |
|---|---:|
| fp16 | 0.602 |
| quantvla_w4a8_atmohb | 0.579 |
| gdsq_vla_atmohb | 0.584 |
| gdsq_vla | 0.559 |

## Efficiency

| Config | Episode wall (s) | Inference/replan (s) | Server infer (ms) | Peak server CUDA MiB | Mean GPU util |
|---|---:|---:|---:|---:|---:|
| fp16 | 38.7 | 0.218 | 196.1 | nan | nan% |
| quantvla_w4a8_atmohb | 44.5 | 0.360 | 326.3 | nan | nan% |
| gdsq_vla_atmohb | 42.1 | 0.295 | 264.4 | nan | nan% |
| gdsq_vla | 112.7 | 0.293 | 255.8 | 20108 | 24.0% |

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
