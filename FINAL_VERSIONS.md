# GDSQ-VLA Final Paper Versions (frozen)

Branch `main` (merged from `codex/d-pac-softfold`), 2026-08-29.
Both model lines are FROZEN as recorded here; every artifact is
content-addressed and referenced by sha256.

## GR00T N1.5 — full-context v2 (formal Table-1 superiority)

- Frozen plan: `runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json`
  - sha256 `969c5ae8bc84719229c81a88e8b6ace745cbb87cea551528ec8cf1fd3820e3a2`
  - mask: 100 W4 + 16 FP16 (historical main mask), common runtime
    (Hessian group-64, row_rotation=0, dynamic A8 DyRange v5, no corrections)
  - total static bytes: 962,068,480 (0.998x QuantVLA cell; <= 1.10x budget)
- v2 quick gate: `runs/full_context_v2/quick/aggregate.json`
  - sha256 `33d0f78972408139608f38d7baf97cfb4bab3efe56054c44a2eecce90477c39f`
  - candidate 28/50 vs gdsq_main 21/50, W=12 L=5 (seeds 60-69, hash-drawn 2/2/1 tasks)
- Table-1 formal aggregate: `runs/full_context_v2/table1/aggregate.json`
  - sha256 `cbb59547a6149456f9ae8fc0bac0fedea1267e5ff5f4d144448112f15b0f459e`
  - candidate 1350/2500 = 54.0% (macro 0.540; splits 74.7/43.9/40.9)
  - vs gdsq_vla_main (50.8%): W=370 L=289, McNemar p=0.0018, Holm 0.0036,
    task-then-seed hierarchical bootstrap 95% CI (0.0048, 0.0604)
    -> **formal_superiority = True**
  - vs fp16 (55.1%): W=292 L=319, p=0.2929, CI (-0.0388, 0.0176) — tied with teacher
  - vs quantvla_w4a8 (30.4%): W=721 L=132, p<1e-4, CI (0.1912, 0.2800)
- Activation attribution: `runs/full_context_v2/p2/activation_attribution.json`
  - sha256 `3c59f27bf648b575bf87864ba9ed4a44590d077d8a57c298d841e6ff4bd879b9`
  - selected mode dynamic_a8; static reference refuted offline (objective 467);
    dynamic vs FP16 control objective 0.52.
- Attribution (H/M/C/C16, dev tasks): `runs/full_context_v2/attribution_interpretation.json`
  - sha256 `874f553d932ceba3c2d11f9ac7e19955ed08f248d3f575fd4bbaa62fbc8ebfcf`
  - H=31 > M=25 (W7/L1); M=25 ~ C=26; C16=26 ~ C=26 (A8 exonerated); proxy alive.

## pi0.5 — full-context v2 (compression-parity non-inferiority anchor)

- Frozen plan: `runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json`
  - sha256 `75d03ac67f17e44570055b54b1a25e44b88ee22bc2282d91bda5acc1f1c61b98`
  - mask: 121 W4 + 59 FP16 (main mask pruned to the 1.10x QuantVLA budget;
    41 lowest-benefit FP16 layers flipped to W4), common runtime
  - total static bytes: 1,634,828,288 (1.097x QuantVLA cell; <= 1.10x budget;
    0.886x exact-main bytes); 2.702x compression vs FP16 (floor 2.691x)
- Combined quick (v1 seeds 50-59 + expansion 60-69, 100 episodes/config):
  `runs/full_context_v2/pi05_quick/combined_aggregate.json`
  - sha256 `9633554be93cdfd88448b8e2ebec8e0999e26a7352685176e20ee2eb6d3fbade`
  - main 33/100 vs candidate 29/100, W=7 L=11 -> not superior
- Non-inferiority anchor: `runs/full_context_v2/pi05_quick/non_inferiority_anchor.json`
  - sha256 `b7922b77071480e5d7d5b37e0492ad22bb42fac23c25f83f22ff872fb190f7b8`
  - status `not_superior_to_gdsq_main`; claim guard: success-parity with many
    more W4 layers (121 vs main's 80) is compression parity, not a
    success-rate improvement.

## Claims

- GR00T N1.5: at QuantVLA-class bytes, the historical main mask on the common
  runtime (Hessian g64 + dynamic A8) formally beats GDSQ-VLA main (+3.2pp,
  p=0.0036 Holm) and ties the FP16 teacher.
- pi0.5: the budget-pruned main mask matches main at higher compression
  (2.702x vs FP16; 1.097x QuantVLA cell) — registered as a non-inferiority
  anchor, explicitly NOT a success-rate claim.
