# DyPAC-VLA ICLR 2027 draft

This directory contains the anonymous ICLR 2027 review draft for **DyPAC-VLA: Full-Context
Mixed-Precision Quantization for Vision-Language-Action Models**. The directory name is retained
as a legacy integration path for existing build and Overleaf automation; it is not the paper name.

The manuscript uses the official ICLR 2027 author kit. The downloaded ZIP has SHA-256
`0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.
The unmodified style and bibliography files have SHA-256 values
`797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6` and
`2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5`.

Build and audit with:

```bash
make check
```

The check first verifies frozen DyPAC evidence and generated table cells, then compiles the PDF and
enforces anonymity, an exact nine-page main text, statement placement, official template hashes,
readable booktabs tables, and a warning-free log.

Active paper artifacts:

- `figures/pipeline_figure.tex`: code-native DyPAC-VLA architecture figure.
- `figures/fcp_algorithm.tex`: fail-closed FCP selection and freeze algorithm.
- `tables/main_results.tex`: GR00T and $\pi_{0.5}$ Table 1, with 54.0% as the formal ours result and the $\pi_{0.5}$ quick result explicitly isolated as a compression anchor.
- `tables/core_ablation.tex`: audited core mechanism evidence in main-text Table 2.
- `tables/core_rollout_plan.tex`: main-text Table 3 with the three preregistered closed-loop causal contrasts.
- `tables/extended_ablation_plan.tex`: appendix-only secondary ablation protocol; pending.
- `tables/omega_qvla_libero.tex`: appendix-only pending LIBERO extension; claim-disabled.
- `dypac_evidence_registry.json`: paper identity, claim guards, source paths, and frozen hashes.

Older GDSQ figures and tables are retained only as provenance and are not included by the active
manuscript.
