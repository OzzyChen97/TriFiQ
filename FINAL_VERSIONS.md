# DyPAC-VLA Final Paper Versions (frozen)

Branch `main` (merged from `codex/d-pac-softfold`), frozen 2026-08-29 and
paper-renamed 2026-08-30. Both model lines are FROZEN as recorded here; every
artifact is content-addressed and referenced by sha256. The active paper does
not compare against an earlier within-project version: GR00T 54.0% is ours,
and Table 1 uses only FP16, QuantVLA, Uniform W6, and $\Omega$-QVLA baselines.

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
  - the frozen aggregate also retains historical internal comparisons for
    provenance, but they are excluded from the active manuscript and claims
- Activation attribution: `runs/full_context_v2/p2/activation_attribution.json`
  - sha256 `3c59f27bf648b575bf87864ba9ed4a44590d077d8a57c298d841e6ff4bd879b9`
  - selected mode dynamic_a8; static reference refuted offline (objective 467);
    dynamic vs FP16 control objective 0.52.
- Attribution (H/M/C/C16, dev tasks): `runs/full_context_v2/attribution_interpretation.json`
  - sha256 `874f553d932ceba3c2d11f9ac7e19955ed08f248d3f575fd4bbaa62fbc8ebfcf`
  - H=31 > M=25 (W7/L1); M=25 ~ C=26; C16=26 ~ C=26 (A8 exonerated); proxy alive.

## pi0.5 — DyPAC-VLA (formal Table-1 result)

- Frozen plan: `runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json`
  - sha256 `e502f7cd7d126517c000f8b5b8e7c6e5537b31226910a2dd6834caaebf736c83`
  - mask: 121 W4 + 59 FP16 (main mask pruned to the 1.10x QuantVLA budget;
    41 lowest-benefit FP16 layers flipped to W4), common runtime
  - total static bytes: 1,634,828,288 (1.097x QuantVLA cell; <= 1.10x budget;
    0.886x exact-main bytes); 2.702x compression vs FP16 (floor 2.691x)
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
  teacher is not significant. FCP validates the existing mask as a local
  optimum; it does not discover a different GR00T mask.
- pi0.5: the budget-pruned main mask reaches 27.7% RoboCasa365 success at
  2.702x compression versus FP16 (1.097x the QuantVLA byte cell) under the
  protocol-matched four-flow-step evaluation. Cross-row gaps are descriptive
  until the registered paired significance analysis is complete.
