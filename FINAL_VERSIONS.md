# DyPAC-VLA Final Paper Versions (frozen)

Branch `main` (merged from `codex/d-pac-softfold`), frozen 2026-08-29 and
paper-renamed 2026-08-30. Both model lines are FROZEN as recorded here; every
artifact is content-addressed and referenced by sha256. The active paper uses
the earlier GDSQ-VLA deployment only as an initializer-stage, identical-mask
control for the combined post-allocation stack; GR00T 54.0% remains ours.
ActQuant and DA-PTQ both have complete 2,500-episode rows for both models; DA-PTQ is
reported as a source-nonequivalent clean-room adaptation. QVLA-code is omitted from Table 1
and its failed public-release reproduction is disclosed in the appendix.

## GR00T N1.5 — DyPAC-VLA (formal Table-1 result)

- Frozen plan: `runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json`
  - sha256 `969c5ae8bc84719229c81a88e8b6ace745cbb87cea551528ec8cf1fd3820e3a2`
  - mask: 100 W4 + 16 FP16 (historical main mask), common runtime
    (Hessian group-64, row_rotation=0, dynamic A8 DyRange v5, no corrections)
  - total static bytes: 962,068,480 (0.998x QuantVLA cell; <= 1.10x budget)
- v2 quick gate: `runs/full_context_v2/quick/aggregate.json`
  - sha256 `33d0f78972408139608f38d7baf97cfb4bab3efe56054c44a2eecce90477c39f`
  - candidate 28/50 vs historical internal reference 21/50, W=12 L=5
    (seeds 60-69, hash-drawn 2/2/1 tasks); not a paper comparison
- Table-1 formal aggregate: `runs/full_context_v2/table1/aggregate.json`
  - sha256 `cbb59547a6149456f9ae8fc0bac0fedea1267e5ff5f4d144448112f15b0f459e`
  - candidate 1350/2500 = 54.0% (macro 0.540; splits 74.7/43.9/40.9)
  - vs fp16 (55.1%): W=292 L=319, p=0.2929, CI (-0.0388, 0.0176) — difference not significant; no equivalence claim
  - vs quantvla_w4a8 (30.4%): W=721 L=132, p<1e-4, CI (0.1912, 0.2800)
  - versus the initializer-stage GDSQ-VLA deployment (50.76%) with the exact
    same 100-W4/16-FP16 mask: W=370, L=289, Holm p=0.003619. This is a combined comparison
    of Hessian-aware W4 + dynamic A8 against original DuQuant W4 +
    split-calibrated static A8, not an FCP mask-selection gain.
- Activation attribution: `runs/full_context_v2/p2/activation_attribution.json`
  - sha256 `3c59f27bf648b575bf87864ba9ed4a44590d077d8a57c298d841e6ff4bd879b9`
  - selected mode dynamic_a8; this artifact preserves the original pre-fix
    selection trace and is superseded for task-level uncertainty statistics by
    the correction record below.
- Corrected selection statistics:
  `runs/full_context_v2/statistics_correction/corrected_statistics.json`
  - jackknife SE of the task mean is
    `sqrt(sum((x-mean)^2)/(n*(n-1)))`; the prior implementation inflated it by
    `n-1`.
  - 2/116 local flips are positive under both D_func and D_PAC; the five
    previously executed complete-policy alternatives remain ineligible with
    corrected objectives 1.54--3.00.
  - static-vs-A16 objective 264.94; dynamic-vs-A16 0.066;
    dynamic-vs-registered-static -0.241.
  - all six corrected candidates completed teacher-state complete-policy
    scoring; 0/6 are eligible, their objectives span 0.948--2.360, and every
    candidate fails at least one component guard. The fail-closed decision
    retains `context_base`, so candidate-state scoring is not triggered.
  - frozen selection report:
    `runs/full_context_v2/fcp_completion/gr00t_corrected_fcp_frozen.json.selection.json`,
    sha256 `7e7830c749d501b516c82ead5e416437e4408ba519914a5b47d0f29613914987`.
- Preregistered post-selection FCP closed-loop diagnostic:
  `runs/full_context_v2/fcp_completion/closed_loop/aggregate.json`, sha256
  `1cedce9f204bd3f265cbf1d604020253c78f89791655b914599582d8891f3c77`.
  - all six candidates complete 50 tasks with ten seeds per task: 3,000 new
    episodes, plus the paired 500-episode frozen `context_base` reference.
  - candidate task-macro success spans 50.6--56.2% versus 53.8% for the
    reference. The largest observed difference is +2.4 points (`ff_pair_0`),
    with raw p=0.285 and Holm p=1.000; the smallest adjusted p in the fixed
    family is 0.891.
  - the decision and test family were frozen before outcome access, and result
    feedback was forbidden. The diagnostic does not revise the retained mask;
    non-significance is not interpreted as equivalence.
- Real A40 hardware measurement:
  `runs/full_context_v2/fcp_completion/hardware/summary.json`, sha256
  `fa7c3e91ca9f678aae3a33882106ff7ab0bb3daaa0dd37a6faaa58715fdc45e5`.
  - nine configurations each use three trials and 180 measured requests.
  - `context_base` versus native FP16: median latency 232.2 versus 107.5 ms,
    median CUDA peak allocation 4.267 versus 5.386 GiB, and gross board energy
    39.81 versus 18.65 J/request. This establishes a measured memory saving but
    not a latency or energy saving for the current generic packed-W4 backend.
- Appendix-only real-integer backend audit:
  `docs/gdsq_vla_iclr2027/evidence/integer_w4a8_backend_audit.json`, sha256
  `8e12ed20550dff16f999af2383a5ae89c49e2d48f17b1deb04ad9a104447c219`.
  - Runtime contract: packed W4 residency with dynamic signed A8 and
    INT8 x INT8 -> INT32 GEMM; kernel compatibility changes activation scaling
    from per-input-channel to per-row group-64.
  - Closed-loop safety rerun: 500/500 paired task-seed keys complete without
    crash, missing, or duplicate rows. The result does not establish
    equivalence or non-inferiority and does not replace the formal Ours row.
  - Paired A40 wave: p50 229.7 -> 187.5 ms, p95 239.4 -> 199.3 ms, gross
    board energy 39.25 -> 34.64 J/request, idle-adjusted energy 22.65 ->
    20.97 J/request, and unchanged 4.267-GiB peak CUDA allocation.
  - The appendix hardware table reports latency, energy, and memory only; it
    deliberately omits success-rate and delta-success columns.
- Attribution (H/M/C/C16, dev tasks): `runs/full_context_v2/attribution_interpretation.json`
  - sha256 `874f553d932ceba3c2d11f9ac7e19955ed08f248d3f575fd4bbaa62fbc8ebfcf`
  - H=31 > M=25 (W7/L1); M=25 ~ C=26; C16=26 ~ C=26 (A8 exonerated); proxy alive.
- Formal-scale maximal-rate rows (secondary, displayed in Table 1 as "Ours (max rate)"):
  - GR00T plan: `runs/robocasa365_table1_max_sweep_v1/masks/gr00t_max.plan.json`
    - sha256 `b1fe584404184bf6f5505af5ef40ebaeab759a94c1e0e952dd19b085e387179d`
    - mask: 116 W4 + 0 FP16 (all-W4 endpoint of the preregistered rate family),
      dynamic A8, static 836,960,256 bytes (2.556x; 0.868x QuantVLA cell)
    - 2,500/2,500 episodes: 1319 successes, task-macro 52.8% (70.9/42.9/42.3 by split);
      cross-run vs the audited 54.0% anchor: 319/288 discordant, McNemar p=0.2233
  - pi0.5 plan: `runs/robocasa365_table1_max_sweep_v1/masks/pi05_max.plan.json`
    - sha256 `8b0aabc15e48970250e605ca689e08bde3c9b73616d2c31fd2fae88d01759dbd`
    - mask: 180 W4 + 0 FP16 (all-180-layer W4 profile), dynamic A8,
      static 1,242,169,344 bytes (3.556x)
    - 2,500/2,500 episodes: 670 successes, task-macro 26.8% (58.7/14.5/3.3 by split);
      cross-run vs the audited 27.7% anchor: 182/159 discordant, McNemar p=0.2335
  - combined aggregate: `runs/robocasa365_table1_max_sweep_v1/aggregate.json`
    - sha256 `93bb7b021ca83b073b314c6b18e97f3d7b4c4e327eeb190375d7d5fedb3070bc`
  - both rows are descriptive operating points, not registered tests; neither
    revises the frozen headline rows (GR00T 54.0% audited anchor and pi0.5 27.7%
    projected anchor remain the formal Table-1 results).

## pi0.5 — DyPAC-VLA (formal Table-1 result)

- Frozen plan: `runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json`
  - sha256 `e502f7cd7d126517c000f8b5b8e7c6e5537b31226910a2dd6834caaebf736c83`
  - mask: 121 W4 + 59 FP16 (main mask pruned to the 1.10x QuantVLA budget;
    41 lowest-benefit FP16 layers flipped to W4), common runtime
  - total static bytes: 1,634,828,288 (1.097x QuantVLA cell; <= 1.10x budget;
    0.886x exact-main bytes); 2.702x compression vs FP16 (floor 2.691x)
- FCP mask adaptation:
  - transferred initializer: 80 W4 + 100 FP16 at 1,845,100,544 bytes, above
    the 1,639,513,497-byte deployment ceiling
  - feasibility projection: 41 least-damaging FP16 protections are changed to
    W4 using paired complete-policy coordinate scores, yielding 121 W4 + 59 FP16
  - post-projection acceptance: both complete-policy proposals fail the minimax
    and component guards, so the frozen report retains the projected anchor
  - projection artifact: `runs/full_context_v2/pi05_p2/pi05_main_pruned_to_budget.json`
    (sha256 `e06870a2f83c6f957df1733dad0d461fcdd0fc66be929035699915d908ce6bcc`)
- Preregistered same-protocol FCP diagnostic:
  - raw aggregate: `runs/full_context_v2/pi05_fcp_diagnostic/raw_aggregate.json`,
    sha256 `7fb09f82d1fbe51ba4ee1af0f1a4d9671f2cce9255882951609242247630601c`;
    experiment summary: `runs/full_context_v2/pi05_fcp_diagnostic/experiment_summary.json`,
    sha256 `6fdcdf4c53f86ff45cf00cee4fe50abbc9c3ad8ceeab5005dbee531d1f19e87e`.
  - exact coverage is 50 tasks x seeds 0--9 for every configuration: 1,500
    new episodes plus the reused 500-episode projected anchor.
  - projected anchor versus transferred initializer: 28.2% versus 26.0%,
    +2.2 points, paired W/L/T 41/30/429, exact McNemar p=0.2351.
  - single-best/two-best proposals: 24.8%/25.0%, or -3.4/-3.2 points
    versus the anchor; both Holm-adjusted p=0.0859.
  - this is a post-selection mechanism diagnostic. It cannot revise the frozen
    choice; the projection result is not claimed as a gain or non-inferiority.
- Formal protocol: four flow steps, execute/replan horizon 16, target split,
  50 seeds per task, paired deterministic action noise, and fresh rendered
  environments under the official horizons.
- Formal aggregate: `runs/full_context_v2/pi05_table1/aggregate.json`
  - sha256 `512480a2e0836423254217b5215489bcb218e17a4c7d0fff1cdf5aba9acf4f73`
  - 693/2,500 = 27.7% task-macro success; splits 59.6/15.4/4.3
  - the protocol metadata correction from 10 to 4 flow steps changes no
    episode outcome, coverage key, timing observation, or storage value
  - correction audit: `runs/full_context_v2/pi05_table1/protocol_correction.json`,
    sha256 `cd5e07baaaf7b40e53c1ace9a881eb49aabb9407d8d20189d46e3a11b2186770`
- Formal Table-1 baselines, all with 2,500 target-split episodes:
  - FP16 and QuantVLA: `runs/pi05_gdsq_gr00t_aligned/official_target_paired50/aggregate/summary.json`
  - Uniform W6: `runs/gdsq_week1_preregistered_v1/execution/runs/pi05_uniform_w6_official50/aggregate/summary.json`
  - $\Omega$-QVLA: `runs/gdsq_extension_preregistered_v1/omega_qvla_pi05_robocasa365_v1/aggregate/summary.json`
  - all completed $\pi_{0.5}$ rows use four flow steps; Ours is 1.6 points
    above FP16 and 2.9 points above QuantVLA W4A8, reported descriptively
    until the paired significance family is registered

## Claims

- Paper identity: **DyPAC-VLA**, expanded as *Dynamic-Range and
  Prefix-Accumulated Control-aware Quantization for Vision-Language-Action
  Models*. The three named components are $D_{PAC}$, Full-Context Precision
  Protection (FCP), and DyRange-A8.
- GR00T N1.5: at QuantVLA-class bytes, DyPAC-VLA uses the historical 100-W4 /
  16-FP16 mask on the common Hessian-g64 + dynamic-A8 runtime. It formally
  improves over QuantVLA by 23.6 points and its difference from the FP16
  teacher is not significant. Under corrected statistics, FCP exposes two
  locally favorable flips but accepts none of the six resulting corrected
  complete-policy candidates; the frozen initializer is retained.
- pi0.5: FCP changes the over-budget 80-W4/100-FP16 transferred initializer
  into a 121-W4/59-FP16 feasible anchor, then abstains from both later
  proposals. The frozen anchor reaches 27.7% RoboCasa365 success at 2.702x
  compression versus FP16 (1.097x the QuantVLA byte cell) under the
  protocol-matched four-flow-step evaluation. Cross-row gaps are descriptive.

## DA-PTQ RoboCasa365 Table-1 extension

- Aggregate: `runs/daptq_table1/formal/aggregate.json` (sha256
  `2c857eab542035650a22bfd54a890fed49afd084e999ee5f17c37adcddf9eeb8`).
- Coverage: 5,000/5,000 new formal episodes, with 50 tasks and seeds 0--49
  for each of GR00T and pi0.5 under the shared four-flow-step protocol.
- GR00T: 76/2,500 = 3.0% (Atomic/C-Seen/C-Unseen: 8.1/0.0/0.4%),
  1.110-GiB mean static checkpoint size and 1.80x compression.
- pi0.5: 655/2,500 = 26.2% (58.6/12.3/3.8%), 1.390 GiB and 2.96x
  compression.
- The registered comparisons against DyPAC-VLA give paired wins/losses of
  8/1,282 on GR00T and 182/220 on pi0.5; the latter has Holm p=0.0648466.
- This is a clean-room cross-architecture adaptation with
  `source_protocol_equivalent=false`; source-reported CogACT/SimplerEnv
  measurements remain separate.

<!-- BEGIN QVLA_ACTQUANT_TABLE1 -->
## ActQuant RoboCasa365 Table-1 extension

- Aggregate: `runs/qvla_actquant_table1/formal/aggregate.json` (sha256 `9f571ac06de8b748d7fdb550c200699c719fc1755a9aaa9b3bf08608491c8484`).
- Calibration: `fp16_teacher_proxy`; `source_protocol_equivalent=false`; formal seeds 0--49 were excluded from calibration.
- ActQuant follows the pinned official GitHub implementation for its allocation and quantization semantics; local code supplies the GR00T/pi0.5 and RoboCasa365 adapters.
- The frozen ActQuant service source is archived under `runs/qvla_actquant_table1/frozen_sources/` and verified against the implementation hash recorded before rollout.
- Coverage: two new rows, each 50 tasks × 50 paired seeds = 2,500 episodes; 5,000/5,000 new formal episodes total.
- QVLA-code is omitted from Table 1 after the GitHub-based RoboCasa365 cross-architecture reproduction produced zero successes before completion; the public release lacks the target architecture/benchmark paths, so the appendix records that incomplete open sourcing cannot be distinguished from an error in the released code path.
- Reported size is measured static pack/GGUF storage only. No PyTorch runtime-memory, C++ runtime, or latency claim is made.

| Method/model | Task-macro SR | Micro SR | Atomic | C-Seen | C-Unseen | Static GiB | Compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| actquant / gr00t | 52.88% | 52.88% | 72.11% | 42.62% | 41.50% | 3.655 | 1.388× |
| actquant / pi05 | 25.16% | 25.16% | 56.56% | 12.25% | 2.75% | 2.904 | 2.154× |

The registered paired family contains ActQuant versus the protocol-compatible DyPAC row for each model (four flow steps for both GR00T and pi0.5; two exact McNemar tests and Holm correction).
<!-- END QVLA_ACTQUANT_TABLE1 -->
