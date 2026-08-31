# Quantization Changes the Data — ICLR 2027 draft

This directory contains the anonymous ICLR 2027 review draft for **Quantization Changes the Data:
Full-Context Post-Training Quantization for Vision-Language-Action Policies**. The directory name is retained
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

- `figures/dypac_main_architecture.pdf`: active vector Figure 1 contrasting fixed-data PTQ, the quantization--control feedback loop, and the full-context response.
- `figures/dypac_main_architecture_figure.tex`: Figure 1 wrapper, caption, and label.
- `scripts/tools/render_dypac_method_figure.py`: deterministic Figure 1 renderer.
- `figures/dypac_fcp_audit.pdf`: active evidence Figure 2 for the GR00T full-policy audit and its abstention outcome.
- `figures/dypac_fcp_audit_figure.tex`: Figure 2 wrapper, caption, and label.
- `figures/dypac_evidence_figures.audit.json`: frozen-source audit for the evidence figures.
- `tables/main_results.tex`: GR00T and $\pi_{0.5}$ Table 1, including the complete 2,500-episode, four-flow-step $\pi_{0.5}$ DyPAC-VLA result.
- `tables/core_ablation.tex`: audited core mechanism evidence in main-text Table 2.
- `tables/core_rollout_plan.tex`: main-text Table 3 with the three preregistered closed-loop causal contrasts.
- `tables/extended_ablation_plan.tex`: appendix-only secondary ablation protocol; pending.
- `tables/libero_transfer_compact.tex`: main-paper LIBERO transfer summary over GR00T N1.5 and $\pi_{0.5}$.
- `tables/omega_qvla_libero.tex`: appendix LIBERO suite breakdown with 4,000/4,000 locally audited episodes, all 40 suite cells complete, and audited static size for every local configuration.
- `dypac_evidence_registry.json`: paper identity, claim guards, source paths, and frozen hashes.

Only assets referenced by the active manuscript, their generation provenance, and evidence audits
are retained in this directory.
