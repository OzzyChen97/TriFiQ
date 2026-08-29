# DyPAC-VLA Final Paper Versions (frozen)

Branch `main` (merged from `codex/d-pac-softfold`), frozen 2026-08-29 and
paper-renamed 2026-08-30. The previous GDSQ-VLA label is retained below only
for the paired historical baseline. Both model lines are FROZEN as recorded
here; every artifact is content-addressed and referenced by sha256.

## GR00T N1.5 — DyPAC-VLA (formal Table-1 result)

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

## pi0.5 — DyPAC-VLA compression anchor

- Frozen plan: `runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json`
  - sha256 `75d03ac67f17e44570055b54b1a25e44b88ee22bc2282d91bda5acc1f1c61b98`
  - mask: 121 W4 + 59 FP16 (main mask pruned to the 1.10x QuantVLA budget;
    41 lowest-benefit FP16 layers flipped to W4), common runtime
  - total static bytes: 1,634,828,288 (1.097x QuantVLA cell; <= 1.10x budget;
    0.886x exact-main bytes); 2.702x compression vs FP16 (floor 2.691x)
- Combined quick (v1 seeds 50-59 + expansion 60-69, 100 episodes/config):
  `runs/full_context_v2/pi05_quick/combined_aggregate.json`
  - sha256 `9633554be93cdfd88448b8e2ebec8e0999e26a7351520752c20ee2eb6d3fbade`
  - main 33/100 vs candidate 29/100, W=7 L=11 -> not superior
- Non-inferiority anchor: `runs/full_context_v2/pi05_quick/non_inferiority_anchor.json`
  - sha256 `b7922b77071480e5d7d5b37e0492ad22bb42fac23c25f83f22ff872fb190f7b8`
  - status `not_superior_to_gdsq_main`; claim guard: the many-more-W4 result
    (121 vs main's 80) is a compression anchor, not success parity or a
    success-rate improvement.

## Claims

- Paper identity: **DyPAC-VLA**, expanded as *Dynamic-Range and
  Prefix-Accumulated Control-aware Quantization for Vision-Language-Action
  Models*. The three named components are $D_{PAC}$, Full-Context Precision
  Protection (FCP), and DyRange-A8.
- GR00T N1.5: at QuantVLA-class bytes, DyPAC-VLA uses the historical 100-W4 /
  16-FP16 mask on the common Hessian-g64 + dynamic-A8 runtime. It formally
  beats the previous GDSQ-VLA runtime (+3.2pp, Holm $p=0.0036$) and is
  statistically tied with the FP16 teacher. FCP validates the existing mask
  as a local optimum; it does not discover a different GR00T mask.
- pi0.5: the budget-pruned main mask reaches 2.702x compression versus FP16
  (1.097x the QuantVLA byte cell), but its 29/100 quick result does not exceed
  the previous main plan's 33/100. This is explicitly a compression anchor,
  not a success-rate or formal non-inferiority claim.
