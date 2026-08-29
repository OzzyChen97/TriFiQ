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
enforces anonymity, the nine-page main-text limit, statement placement, official template hashes,
readable booktabs tables, and a warning-free log.

Active paper artifacts:

- `figures/pipeline_figure.tex`: code-native DyPAC-VLA architecture figure.
- `tables/main_results.tex`: audited GR00T table with 54.0% as ours.
- `tables/pi05_anchor.tex`: compression-only $\pi_{0.5}$ boundary.
- `dypac_evidence_registry.json`: paper identity, claim guards, source paths, and frozen hashes.

Older GDSQ figures and tables are retained only as provenance and are not included by the active
manuscript.
