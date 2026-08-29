# D-PAC + SoftFold architecture figures (branch `codex/d-pac-softfold`)

Generated with `gpt-image-2` (quality: high) through the Images API in
`config.txt` (endpoint `https://di2lab.zeabur.app/v1/images/generations`),
using `scripts/tools/generate_gdsq_architecture_candidate.py`.
Each PNG has a `.png.json` sidecar with endpoint, prompt SHA256 and output SHA256.

| File | Content |
|---|---|
| `01_main_architecture.png` | End-to-end GDSQ-VLA pipeline: isolated layer probing → dual similarity + D-PAC scoring → byte-constrained mask search → complete-configuration functional adjudication (d-func) → calibration-gated runtime selector with SoftFold 9×9 gate. Deployed output (M, v): GR00T 100 W4A8 + 16 FP16, 1.99× storage, 50.8% success. |
| `02_contribution1_allocation.png` | **Contribution 1** — training-free mixed-precision W4/FP16 allocation: 1−CKA geometry + D_CS distribution, D-PAC action weighting w_i = n·d_i/Σd_j, RMS/saturation stability guards, frozen byte budget. |
| `03_contribution2_adjudication.png` | **Contribution 2** — complete-configuration functional adjudication: Top-K masks, paired observations and identical noise, d-func = D_final + D_kin + D_grip + 2·CVaR0.9(D_solver), frozen 16:1 rule, 10k bootstrap plateau. |
| `04_contribution3_selector.png` | **Contribution 3** — calibration-gated runtime correction selector: ATM / OHB / baseline candidate rule + v8 absolute gate (q_dir ≥ 0.30, q_non ≥ 0.85, q_log ≥ 0.05), frozen SHA, no task identity or rollout feedback. GR00T → baseline, π0.5 → OHB. |
| `05_dpac_softfold_closeup.png` | Branch-core close-up: D-PAC paired physical-action consistency metric (local / prefix / SE(3) pose / stitch / gripper, robust scale, CVaR0.9) and SoftFold 9×9 soft-gated ATM × ErrorFold compensation grid. |

Prompts are stored under `prompts/`. Note: the API honored the requested
2048×1152 aspect but returned 1672×941 rasters.
