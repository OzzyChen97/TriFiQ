# DyPAC-VLA adversarial review notes

Review date: 2026-08-30. This file is internal and is not included in the submission PDF.

## Reverse outline

- **Abstract:** control-error accumulation challenge; $D_{PAC}$, FCP, and DyRange-A8; GR00T
  54.0% headline; explicit $\pi_{0.5}$ claim boundary.
- **Introduction:** why local PTQ metrics and static ranges fail; full-context insight; the exact
  source of the gain; three contributions.
- **Related work:** VLA efficiency, VLA PTQ, general PTQ substrate, and mixed-precision behavioral
  sensitivity; every paragraph ends with the technical distinction.
- **Method:** paired physical action metric; exact-budget counterfactual search and whole-network
  adjudication; Hessian group-64 W4 and per-forward channelwise A8.
- **Experiments:** paired 50-task protocol; GR00T headline; runtime-vs-mask attribution; separate
  $\pi_{0.5}$ compression boundary.
- **Conclusion:** result, mechanism, and three scope limitations.

## Claim–evidence map

| Claim | Frozen evidence | Status |
|---|---|---|
| DyPAC-VLA obtains 54.0% GR00T SR | `runs/full_context_v2/table1/aggregate.json`, 1350/2500 | supported |
| +3.2pp over GDSQ-VLA, Holm $p=0.0036$ | same aggregate, W=370/L=289, CI [0.0048, 0.0604] | formal superiority |
| Statistically tied with FP16 55.1% | same aggregate, W=292/L=319, $p=0.2929$, CI [-0.0388, 0.0176] | supported as non-significant difference; no equivalence claim |
| 2.22× GR00T static-component compression | frozen plan, 962,068,480 / 2,139,537,408 bytes | supported; static storage only |
| FCP discovers a new GR00T mask | decision record returns the historical 100-W4/16-FP16 mask | rejected; paper says validation/local optimum |
| DyRange is the principal runtime change | static objective 467; dynamic-vs-FP16 objective 0.52; dev runtime 31 vs 25 | diagnostic support; not isolated formal causal proof |
| $\pi_{0.5}$ improves success | quick screen 29/100 vs 33/100, W=7/L=11 | rejected |
| $\pi_{0.5}$ reaches 2.702× compression | frozen plan, 1,634,828,288 / 4,416,602,112 bytes | compression anchor only |
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
- **Pass:** the GR00T mask/gain distinction is repeated in Introduction, Experiments, Conclusion,
  figure caption, and supplement.

### 3. Experimental strength

- **Pass:** the headline has 2,500 paired episodes and corrected significance against the previous
  runtime.
- **Pass:** absolute SR is statistically indistinguishable from the FP16 teacher at 2.22× static
  compression.
- **Risk disclosed:** the second model does not have a formal success result and is not presented
  as one.

### 4. Evaluation completeness

- **Pass:** FP16, uniform QuantVLA W4A8, and the previous same-mask runtime appear in the main table.
- **Pass:** all three RoboCasa365 task groups and paired uncertainty are reported.
- **Needs new experiment:** a formal $\pi_{0.5}$ matrix and fused-kernel benchmarking would be
  needed for cross-model success and deployment-speed claims.

### 5. Method design soundness

- **Pass:** one global static mask avoids per-task routing and test-time feedback.
- **Pass:** worst-context one-standard-error and component safety rules expose harmful averages.
- **Risk disclosed:** local optimality is limited to the declared flip/proposal family and does not
  establish a global optimum.

## Final checks

- [x] GR00T 54.0% is the only row labeled ours.
- [x] Abstract, Introduction, Results, and Conclusion match the frozen aggregate.
- [x] $\pi_{0.5}$ is a compression anchor, not success parity, equivalence, or non-inferiority.
- [x] No runtime selector/correction is part of the method.
- [x] No latency or live-memory claim is made.
- [x] PDF visually inspected; architecture and result tables are readable.
- [x] Official ICLR style/BST hashes unchanged; anonymous main text ends on page 6.
- [x] No overfull boxes, undefined citations, or undefined references.
