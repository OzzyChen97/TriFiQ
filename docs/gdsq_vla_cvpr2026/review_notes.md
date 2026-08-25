# GDSQ-VLA draft review notes

## Compact paper outline

1. Explain why uniform PTQ is brittle for coupled language and diffusion-action stacks.
2. Introduce dual representation evidence: CKA for geometry and CS divergence for distribution overlap.
3. Convert representation evidence into a deployable mask using action weights, hard guards, a storage budget, and complete-configuration adjudication.
4. Freeze the 16:1 ratio on four declared development tasks.
5. Evaluate on 14 separate atomic tasks, then report the 18-task aggregate, ablations, storage, and runtime boundaries.

## Reverse outline and paragraph roles

### Abstract

- Challenge: VLA layers are heterogeneous under quantization.
- Method: dual similarity, action weighting, guards, and functional adjudication.
- Evidence: primary 14-task result, theoretical storage, and FP16 gap.

### Introduction

- Opening: VLA deployment motivates training-free PTQ.
- Challenge: errors move from multimodal features into the iterative action solver.
- Prior limitation: local metrics expose only one view of damage.
- Insight: complementary representation evidence needs a behavioral referee.
- Method: 116-layer probing and budgeted binary allocation.
- Evidence: primary improvement and deployment caveat.
- Contributions: formulation, procedure, and controlled evaluation.

### Method

- Overview: four-stage execution order.
- Setting: candidate scope and protected attention projections.
- Dual references: attribution versus deployment semantics.
- CKA: geometric preservation and scale blind spot.
- CS: distribution overlap and bandwidth limitation.
- Selection: action weighting, hard guards, and storage budget.
- Adjudication: cross-layer action behavior and ratio freeze.

### Experiments

- Setup: one benchmark, declared dev/primary split, paired protocol, task-level statistics.
- Unified result table: group the official Atomic, Composite-Seen, and Composite-Unseen tasks in one comparison, add their task-weighted 50-task macro Mean, and retain GR00T N1.5 and planned $\pi_{0.5}$ policy blocks.
- Selection control: show the four-task ratio sweep first, then report Primary 14 separately as held-out evidence after tuning is frozen.
- Main result: gain over the fixed W4A8 layout on the held-out GR00T N1.5 Primary 14 tasks.
- Boundary: remaining gap to FP16.
- Ratio: closed-loop SR supersedes proxy ordering.
- Ablations: signal composition and ATM/OHB interaction.
- Runtime: theoretical storage is separated from eager implementation behavior.
- Missing evidence: $\pi_{0.5}$ cells remain TBD until complete frozen matrices are available.

### Conclusion

- Takeaway: geometry, distribution, and action behavior jointly improve allocation.
- Limitations: one model family, binary precision, a remaining FP16 gap, and no fused low-bit runtime.

## Claim--evidence map

- Claim: 16:1 is the frozen ratio. | Evidence: `runs/robocasa365_cs_loss/final_selection_official50.json`, 161/200, 80.5%. | Status: supported.
- Claim: GDSQ-VLA reaches 65.9% on the 14 tasks excluded from ratio selection. | Evidence: recomputation from the frozen official atomic summary after excluding the four `decision.selection_tasks`; 461/700. | Status: supported.
- Claim: GDSQ-VLA improves over QuantVLA W4A8+ATM/OHB by 20.3 points with CI [13.9, 26.9]. | Evidence: paired task-level recomputation on the same 14 tasks; Holm-adjusted p=0.0012. | Status: supported.
- Claim: GDSQ-VLA remains 8.0 points below FP16. | Evidence: paired task-level recomputation; CI [-12.1, -3.9], Holm-adjusted p=0.0073. | Status: supported.
- Claim: the final mask contains 100 W4A8 and 16 FP16 choices among 116 candidates. | Evidence: frozen final-plan layer map. | Status: supported.
- Claim: theoretical LLM+DiT component storage is 1.001 GiB, or 1.99x compression. | Evidence: `paper_style_memory` in the official atomic summary and the checkpoint/plan-based memory calculator. | Status: supported, theoretical only.
- Claim: CS and CKA are universally complementary. | Evidence: only a five-seed diagnostic screen and one checkpoint-specific ratio sweep. | Status: unsupported as a universal claim; paper uses checkpoint-scoped wording.
- Claim: the method transfers to seen composite tasks. | Evidence: complete frozen Composite-Seen summary, 40.6% versus 19.6% for W4A8+ATM/OHB; paired gain 21.0 points with CI [14.3, 28.0]. | Status: supported for Composite-Seen only.
- Claim: the atomic-selected allocation transfers better than the fixed W4A8 layout to unseen composite tasks. | Evidence: complete frozen Composite-Unseen summary, 40.4% versus 19.8%; paired gain 20.6 points with CI [14.5, 26.8] and Holm-adjusted p=0.0002. | Status: supported for the GR00T N1.5 Composite-Unseen split; not lossless versus FP16 (45.3%).
- Claim: the improvement persists across all official RoboCasa365 task groups. | Evidence: complete 50-task aggregate, 50.8% versus 30.4% for W4A8+ATM/OHB; paired gain 20.3 points with CI [16.9, 24.0]. | Status: supported for the frozen GR00T N1.5 checkpoints.
- Claim: GR00T GDSQ-VLA is competitive with the uniform-W6 fixed-ceiling control. | Evidence: the completed 2,500-episode W6 matrix gives 51.7% at 1.109 GiB versus 50.8% at 1.001 GiB for ours; the paired ours-minus-W6 difference is -1.0 point with 95% task CI [-3.1, 1.2]. The replay audit reproduces the W6 summary byte-for-byte. | Status: supported as descriptive storage--accuracy evidence; not a superiority claim.
- Claim: the current implementation accelerates inference or lowers live GPU memory. | Evidence: eager runtime shows the opposite. | Status: rejected; paper explicitly limits the claim to theoretical component storage.
- Claim: static \method transfers to $\pi_{0.5}$ with a significant gain. | Evidence: the complete four-configuration 2,500-episode matrix gives 25.3% for static GDSQ versus 24.8% for the fixed low-bit baseline, but the held-out paired interval crosses zero. | Status: rejected as a superiority claim; reported as an architecture boundary. The formal v8 selector run remains separate until complete.
- Claim: the GR00T $\Omega$-QVLA Atomic result is 60.1\% at a theoretical 0.599 GiB (3.33$\times$). | Evidence: a frozen 900-episode manifest, exact 18-task$\times$50 coverage, zero formal failures, a strict 10,000-draw task-cluster summary, and an independently hashed packed-storage audit. | Status: supported for the Atomic and storage cells only; Composite and all-task claims remain pending.
- Claim: Table 2 will compare local FP16, QuantVLA W4A8, Uniform W6, $\Omega$-QVLA W4A4, and GDSQ-VLA on LIBERO. | Evidence: pinned code/packs and a frozen 4,000-episode joint protocol; no author-reported result values are imported. | Status: pending until the exact local coverage and hash gates complete; it is independent of the completed Table 1 GR00T Atomic subset.

## Frozen provenance

- Ratio selection SHA256: `52f0c15fd24e9eeb0fbaf9edfc51f0f86a54867d8694cc056b01d90b8feb43b6`.
- Official atomic summary SHA256: `9f06db12444b6c582f451c2cc096bcb5ba5738cf888558955acc039aedfed29f`.
- Official Composite-Seen summary SHA256: `372688e3ac4e20eb83f2b752b2bf357d190e91d75b6cb0629e02eaf5296aa58e`.
- Official Composite-Unseen summary SHA256: `147c400a18689d27b625a943123d8948824fd7a9b4f6095fc13a8b4cfa8a5213`.
- Official 50-task aggregate summary SHA256: `243162a79a57fa9bf85402364e2cb6bf50c69baa5320fc2256e5821dfd0ef810`.
- GR00T Uniform-W6 50-task summary SHA256: `3a9165fe10fd69de8c23b254527135c4b30a5d20b8b7bf1030609e887ec75c36`; three-manifest replay audit SHA256: `469440b3f4802af8a82e5ba1661ed8c78f25e4d85e2d5b5f501639d15569b1b0`.
- LIBERO Table 2 five-configuration preregistration SHA256: `1f2bc757bc0adbefa6466a89bfcb24d6b6595628e0cbe0f147fa0cd3af23bf73`.
- Five-seed diagnostic summary SHA256: `da948a0b16976e41a54900305e88ee6e86b95493d8115b871980b3bc2ce0a577`.
- CVPR template tag: `CVPR2026-v1(latex)` at commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`.
- QuantVLA arXiv v4 source SHA256: `a719a574b3ed8a58533c75f2273288a479b900d385db0a6aef47dfca25893228`.
- RoboCasa365 arXiv v1 source SHA256: `f10bbc07c60b72d81332b5d1c796690b87613ab82304565e30daccc600b8e568`.
- Full citation and result-placeholder provenance: `reference_audit.md`.

## Architecture figure provenance

- Generator: `gpt-image-2`, high quality, through the user-supplied OpenAI-compatible Images API; the API key is neither stored in figure metadata nor copied into the paper tree.
- Compared selector-aware candidates: `figures/candidates/gdsq-d-four-stage-selector.png`, `gdsq-e-offline-runtime.png`, `gdsq-f-selector-gate.png`, and `gdsq-g-minimal-evidence.png`; prompts and JSON provenance sidecars are retained.
- User-selected base: candidate D. The final postprocess removes the top title band and corrects the score label from a misleading multiplicative rendering to the paper's additive `Score = w[16(1 - CKA) + D_CS]` rule without regenerating the four content panels.
- Final candidate: `figures/candidates/gdsq-d-four-stage-selector-no-title.png`, SHA256 `e247a63c26dec26b67ec1971a3ffba7ddbeecc745580985281c5f7685819bad6`; its sidecar records the source and deterministic operations.
- Final paper asset: `figures/gdsq_pipeline.png`, normalized to 2048x919 PNG; SHA256 `941507078d05c8e6be6c741e9354849ca7459e9b8d217cace5ac09aee90f0727`.
- Visual audit: the four panels separate intervention evidence, CKA/CS evidence, W6-equivalent constrained masking with guards/adjudication, and a frozen mutually exclusive baseline/ATM/OHB selector. The selector examples match the v8 decisions and explicitly forbid task or rollout feedback.

## Five-dimension adversarial self-review

### 1. Contribution

- Pass: the paper defines a concrete dual-reference, dual-similarity selection pipeline rather than presenting another fixed layout.
- Pass: configuration-level functional adjudication addresses a clear limitation of additive layer proxies.
- Risk: novelty is layer allocation over a known quantization substrate; the paper must emphasize the controlled formulation and behavioral evidence, not claim a new low-bit operator.

### 2. Writing clarity

- Pass: the candidate scope, references, metrics, guards, budget, ratio rule, and final mask are defined.
- Pass: each paragraph has one role and terminology is stable.
- Pass: inspected the generated architecture figure on page 4 of the final review PDF at full-page resolution; labels and arrows remain readable at full paper width.

### 3. Experimental strength

- Pass: primary tasks are separated from ratio-selection tasks.
- Pass: paired uncertainty and corrected significance are reported.
- Risk: the method remains below FP16; this is framed as an explicit compression--accuracy trade-off.

### 4. Evaluation completeness

- Pass: strong fixed-layout, FP16, ratio, signal, and calibration comparisons are present.
- Pass: Atomic, Composite-Seen, and Composite-Unseen matrices are complete and strictly validated for GR00T N1.5.
- Needs new experiment: $\pi_{0.5}$ matrices must be complete before any cross-policy claim.

### 5. Method design soundness

- Pass: hard guards prevent similarity metrics from overriding activation feasibility.
- Pass: the final ratio is selected by closed-loop SR rather than proxy score alone.
- Risk: common random numbers differ from independent native GPU noise; the protocol is labeled paired and not presented as identical to an independent-noise leaderboard setting.

## Final draft checks

- [x] Replace the pipeline placeholder with a validated GPT-Image 2 PNG.
- [x] Replace Composite-Seen TBD cells from its complete frozen summary.
- [x] Replace Composite-Unseen TBD cells only from its complete frozen summary.
- [x] Re-run claim/evidence audit after inserting Composite-Unseen and the complete 50-task aggregate.
- [x] Confirm the main-paper end label is on page 7 (limit: 8), the supplement starts on page 10, and the complete review PDF is 11 pages including references and supplement.
- [x] Confirm no missing citations, references, or anonymization leaks.
- [x] Confirm FP16 is labeled uncompressed and excluded from best-compressed highlighting; \method (Ours) is the final row in direct configuration comparisons.
- [x] Cross-check all 43 cited BibTeX entries; consolidate official RoboCasa365 task groups in one table and report the internal Primary 14 split separately after ratio selection.
- [x] Audit all tabular assets against the official CVPR 2026 author kit (`CVPR2026-v1(latex)`, commit `12909ae...`): captions above with template-controlled small Roman font, `booktabs` rules, no vertical lines, no scaling, `\small` main-table bodies, and no body text below 8pt `\footnotesize`.
- [x] Add $\Omega$-QVLA v1 as a protocol-separated LIBERO table, pin its official code and pack revisions, and keep incomplete local reproduction cells claim-disabled.
