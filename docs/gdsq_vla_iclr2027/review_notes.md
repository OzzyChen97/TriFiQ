# DyPAC-VLA adversarial review notes

Review date: 2026-08-30. This file is internal and is not included in the submission PDF.

## Reverse outline

- **Abstract:** control-error accumulation challenge; $D_{PAC}$, FCP, and DyRange-A8; GR00T
  54.0% headline; explicit $\pi_{0.5}$ claim boundary.
- **Introduction:** why local PTQ metrics and static ranges fail; full-context insight; the exact
  source of the gain; three contributions.
- **Related work:** VLA efficiency, VLA PTQ, general PTQ substrate, and mixed-precision behavioral
  sensitivity; every paragraph ends with the technical distinction.
- **Method:** explicit constrained problem; complete $D_{PAC}$ definition; exact-budget
  counterfactual search, task-level minimax adjudication, and fail-closed freeze algorithm;
  Hessian group-64 W4 and per-forward channelwise A8.
- **Experiments:** paired 50-task protocol; two-model Table 1; GR00T split and task-direction
  analysis; audited mechanism evidence; main closed-loop ablation matrix; separate $\pi_{0.5}$
  compression boundary and failure-accounting rule.
- **Appendix:** secondary diagnostic ablation plan and claim-disabled LIBERO extension.
- **Conclusion:** result, mechanism, and four scope limitations.

## Claim–evidence map

| Claim | Frozen evidence | Status |
|---|---|---|
| DyPAC-VLA obtains 54.0% GR00T SR | `runs/full_context_v2/table1/aggregate.json`, 1350/2500 | supported |
| +23.6pp over QuantVLA W4A8 | same aggregate, W=721/L=132, Holm $p=1.89\times10^{-98}$, CI [0.1912, 0.2800] | formal superiority |
| Difference from FP16 55.1% is not significant | same aggregate, W=292/L=319, $p=0.2929$, CI [-0.0388, 0.0176] | supported; no equivalence claim |
| 2.22× GR00T static-component compression | frozen plan, 962,068,480 / 2,139,537,408 bytes | supported; static storage only |
| FCP discovers a new GR00T mask | decision record returns the initialized 100-W4/16-FP16 mask | rejected; paper says validation/local optimum |
| Static A8 is mismatched; DyRange is closer to the activation control | frozen activation attribution, $J=467.52$ vs $0.518$ | supported as offline mechanism evidence; not closed-loop causal proof |
| Initialized mask is locally retained | 0/116 positive-benefit flips; 0/5 eligible structured alternatives | supported only in the declared search neighborhood |
| $\pi_{0.5}$ improves success | quick screen 29/100 vs 33/100, W=7/L=11 | rejected |
| $\pi_{0.5}$ reaches 2.702× compression | frozen plan, 1,634,828,288 / 4,416,602,112 bytes | compression anchor only |
| Formal $\pi_{0.5}$ baseline cells use 2,500 episodes | frozen FP16/QuantVLA, W6, and $\Omega$-QVLA summaries; exact source hashes in registry | supported; not compared to the 29/100 anchor |
| Current code improves latency/live memory | eager fake-quant execution is not a fused deployment | rejected |

The generated registry `dypac_evidence_registry.json` and
`scripts/tools/render_dypac_paper.py --check` bind every headline cell to the frozen source hashes.

## Five-dimension adversarial self-review

### 1. Contribution

- **Pass:** $D_{PAC}$ adds physical prefix accumulation, SE(3), stitch, gripper timing, and tail
  risk to VLA quantization evaluation.
- **Pass:** FCP makes the whole network—not an additive layer proxy—the final precision judge.
- **Risk disclosed:** Hessian W4 is a substrate, not claimed as a new weight quantizer.

### 2. Writing clarity

- **Pass:** the paper consistently uses DyPAC-VLA, $D_{PAC}$, FCP, and DyRange-A8.
- **Pass:** selection-buffer size, metric terms, budget, acceptance rule, W4 format, and A8 formula
  are specified.
- **Pass:** Figure 1 exposes the evidence--search--deployment chain, exact-byte gate, fail-closed
  freeze, one global mask, and absence of task routing/correction; the generated minimax box was
  corrected to match the frozen adjudication rule rather than an invented formula.
- **Pass:** the GR00T mask evidence boundary is repeated in Introduction, Experiments, Conclusion,
  figure caption, and supplement.

### 3. Experimental strength

- **Pass:** the headline has 2,500 paired episodes and corrected significance against QuantVLA.
- **Pass:** the FP16 difference is reported as non-significant, without an equivalence claim.
- **Risk disclosed:** the second model does not have a formal success result and is not presented
  as one.
- **Risk disclosed:** core closed-loop component-removal rollouts are specified but pending.

### 4. Evaluation completeness

- **Pass:** FP16, QuantVLA W4A8, Uniform W6, $\Omega$-QVLA, and ours appear for both model
  families in the main table, with the non-matching $\pi_{0.5}$ protocol visibly separated.
- **Pass:** all three RoboCasa365 task groups and paired uncertainty are reported.
- **Pass:** important mechanism ablations are in the main text; lower-priority diagnostics and the
  incomplete LIBERO extension are in the appendix.
- **Needs new experiment:** the three planned core rollout contrasts, a formal $\pi_{0.5}$
  matrix, and fused-kernel benchmarking are needed for causal, cross-model, and speed claims.

### 5. Method design soundness

- **Pass:** one global static mask avoids per-task routing and test-time feedback.
- **Pass:** worst-context one-standard-error and component safety rules expose harmful averages.
- **Risk disclosed:** local optimality is limited to the declared flip/proposal family and does not
  establish a global optimum.

## Final checks

- [x] GR00T 54.0% is the only formal row labeled ours; the $\pi_{0.5}$ ours anchor is explicitly non-comparable.
- [x] The active manuscript contains no comparison with the within-project predecessor.
- [x] LIBERO is appendix-only and every cell is marked pending.
- [x] Main-text mechanism evidence is separated from planned-but-unrun ablations.
- [x] Abstract, Introduction, Results, and Conclusion match the frozen aggregate.
- [x] $\pi_{0.5}$ is a compression anchor, not success parity, equivalence, or non-inferiority.
- [x] No runtime selector/correction is part of the method.
- [x] No latency or live-memory claim is made.
- [x] PDF visually inspected; architecture and result tables are readable.
- [x] Official ICLR style/BST hashes unchanged; anonymous main text ends exactly on page 9.
- [x] No overfull boxes, undefined citations, or undefined references.
