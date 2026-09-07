# Quantization Changes the Data — ICLR 2027 draft

This directory contains the anonymous ICLR 2027 review draft for **Quantization Changes the Data:
Closed-Loop-Aware Quantization for Vision-Language-Action Policies**. The directory name is retained
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

- `figures/dypac_main_architecture_figure.tex`: active Figure 1, a readable native-LaTeX schematic contrasting fixed-buffer calibration, the closed-loop dependency, and the offline/deployment response.
- `figures/dypac_main_architecture.pdf` and `scripts/tools/render_dypac_method_figure.py`: retained legacy vector overview and its renderer; not used by the active manuscript.
- `figures/dypac_fcp_audit.pdf`: appendix evidence figure for the corrected GR00T proposal screen and complete-policy gate.
- `figures/dypac_fcp_audit_figure.tex`: evidence-figure wrapper, caption, and label.
- `figures/dypac_evidence_figures.audit.json`: frozen-source audit for the evidence figures.
- `tables/main_results.tex`: GR00T and $\pi_{0.5}$ Table 1, including complete 2,500-episode, four-flow-step DyPAC-VLA, ActQuant, and DA-PTQ rows; the failed QVLA-code reproduction is disclosed only in the appendix.
- `tables/core_ablation.tex`: appendix table with the corrected mechanism audit.
- `tables/core_rollout_plan.tex`: appendix controls with the identical-mask post-initialization contrast, the dynamic-range intervention, and the outcome-blinded six-proposal FCP diagnostic.
- `tables/fcp_candidate_rollouts.tex`: appendix table with all six preregistered FCP proposal outcomes, paired tests, and Holm-adjusted values.
- `tables/pi05_fcp_diagnostic.tex`: appendix table for the preregistered same-protocol $\pi_{0.5}$ projection and post-projection proposal diagnostic (1,500 new episodes plus the reused 500-episode anchor).
- `tables/hardware_results.tex`: appendix-only A40 latency, CUDA peak-allocation, and NVML board-energy table. Its final two rows are a separate paired reference/real-integer backend A/B; success-rate columns are intentionally excluded.
- `evidence/integer_w4a8_backend_audit.json`: content-addressed audit for the real-integer runtime contract, complete 500-episode safety rerun, and paired hardware wave. This evidence does not replace the formal Ours row.
- `tables/predictive_validity.tex`: generated audit artifact for the 60-mask diagnostic; the paper reports the distance-confounding caveat in concise appendix prose rather than embedding this pooled-only table.
- `tables/daptq_source_reported.tex`: source-reported CogACT/SimplerEnv DA-PTQ reference values, separated from the complete, source-nonequivalent clean-room RoboCasa365 comparison in Table 1.
- `tables/omega_qvla_libero.tex`: appendix table with the complete LIBERO suite breakdown, 4,000/4,000 locally audited episodes, all 40 suite cells complete, and audited static size for every local configuration.
- `tables/libero_transfer_compact.tex`: active main-paper compact LIBERO comparison.
- `figures/dypac_operating_points.pdf`: appendix-only GR00T success--storage visualization over selected measured configurations.
- `dypac_evidence_registry.json`: paper identity, claim guards, source paths, and frozen hashes.

The directory retains active manuscript assets, generated alternate views, their provenance, and evidence audits.
