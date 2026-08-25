# Reference audit

Audit date: 2026-08-25. This file is internal and is not included in the submission PDF.

## Acceptance policy

- A citation is retained only when its title and authors match an arXiv record or an official proceedings/publisher record.
- Published venue, year, pages, and DOI fields follow the proceedings/publisher record when available; arXiv identifiers are retained as a second locator.
- arXiv revision year is not substituted for the original publication year. This matters for records such as AWQ, whose current arXiv BibTeX export reflects a later revision.
- `and others` is used only to shorten verified long author lists; it does not stand in for an unknown author list.
- The Semantic Scholar public Graph API returned HTTP 429 during this audit, so no unverified field was imported from it. Metadata instead comes from arXiv plus Crossref, DBLP/OpenReview, CVF, PMLR, or the original paper source.

## Primary paper sources

- QuantVLA: arXiv `2602.20309v4`; source archive SHA256 `a719a574b3ed8a58533c75f2273288a479b900d385db0a6aef47dfca25893228`.
- RoboCasa365: arXiv `2603.04356v1`; source archive SHA256 `f10bbc07c60b72d81332b5d1c796690b87613ab82304565e30daccc600b8e568`.
- $\Omega$-QVLA: exact user-specified arXiv `2605.28803v1`; PDF SHA256 `d0c2ede8a71a497e423c31a2431b0166592f61093acfcfeb6be1be62a7f716e7` (8,510,176 bytes, 18 pages). Table 1 was transcribed into a machine-audited source record rather than copied by hand into LaTeX.
- The QuantVLA source was used to identify the closest technical literature, but every imported record was checked against its own canonical metadata source.

## Canonical arXiv checks

- VLA and robot policies: RT-1 `2212.06817`, PaLM-E `2303.03378`, RT-2 `2307.15818`, Octo `2405.12213`, OpenVLA `2406.09246`, RDT-1B `2410.07864`, $\pi_0$ `2410.24164`, $\pi_{0.5}$ `2504.16054`, GR00T N1 `2503.14734`, SmolVLA `2506.01844`, EfficientVLA `2506.10100`, VLA-Cache `2502.02175`, and MoLe-VLA `2503.20384`.
- Transformer/VLM PTQ: SmoothQuant `2211.10438`, GPTQ `2210.17323`, AWQ `2306.00978`, OmniQuant `2308.13137`, BRECQ `2102.05426`, QuaRot `2404.00456`, DuQuant `2406.01721`, SpinQuant `2405.16406`, FlatQuant `2410.09426`, OstQuant `2501.13987`, Q-VLM `2410.08119`, and MBQ `2412.19509`.
- Diffusion PTQ: PTQ4DM `2211.15736`, Q-Diffusion `2302.04304`, PTQ4DiT `2405.16005`, Q-DiT `2406.17343`, ViDiT-Q `2406.02540`, MixDQ `2405.17873`, and SVDQuant `2411.05007`.
- Similarity and allocation: SVCCA `1706.05806`, PWCCA `1806.05759`, CKA `1905.00414`, HAWQ `1905.03696`, HAWQ-V2 `1911.03852`, HAQ `1811.08886`, and CS-Aligner `2502.17028`.
- Adjacent VLA compression: ActQuant `2605.24011`, $\Omega$-QVLA `2605.28803v1`, and SQAP-VLA `2509.09090`.

## Publisher and proceedings cross-checks

- Diffusion Policy: RSS 2023, DOI `10.15607/RSS.2023.XIX.026`.
- PTQ4DM: CVPR 2023, pages 1972--1981, DOI `10.1109/CVPR52729.2023.00196`.
- Q-Diffusion: ICCV 2023, pages 17489--17499, DOI `10.1109/ICCV51070.2023.01608`.
- PTQ4DiT: NeurIPS 2024, pages 62732--62755, DOI `10.52202/079017-2006`.
- MixDQ: ECCV 2024, pages 285--302, DOI `10.1007/978-3-031-72630-9_17`.
- Q-DiT: CVPR 2025, pages 28306--28315, DOI `10.1109/CVPR52734.2025.02636`.
- MBQ: CVPR 2025, pages 4167--4177, DOI `10.1109/CVPR52734.2025.00394`.
- Accurate PTQ with Small Calibration Sets: PMLR volume 139, pages 4466--4475.
- DBLP/OpenReview records confirm BRECQ (ICLR 2021), SpinQuant (ICLR 2025), and FlatQuant (ICML 2025).

## Unified RoboCasa365 result policy

`tables/main_results.tex` is the unified RoboCasa365 comparison. It groups Atomic, Composite-Seen, and Composite-Unseen columns with their task-weighted 50-task macro Mean, then groups rows by GR00T N1.5 and $\pi_{0.5}$. `tables/primary14_results.tex` separately reports the internal held-out split after the CKA:CS ratio is frozen on four disjoint development tasks. A task-set cell may be populated only when that complete subset has exact task/seed coverage and an audited summary; other cells in the same row remain explicit `pending`. Uneven partial logs and values from different protocols are never inserted.

- The $\pi_{0.5}$ Uniform-W6 control is complete at exactly 2,500 unique episodes (900 Atomic, 800 Composite-Seen, and 800 Composite-Unseen), with no missing or duplicate keys. Its strict summary is `runs/gdsq_week1_preregistered_v1/execution/runs/pi05_uniform_w6_official50/aggregate/summary.json`, SHA256 `849e1e159daf6b515bcb1724a8862fea17e9c0c95053ed5fefc99b9c145e052a`; the frozen manifest SHA256 is `7abe10e3389a362dc118a9f7c50da542f72bdf15d7517f66aa2c8fb3c1cce476`.
- The GR00T Uniform-W6 control is complete at exactly 2,500 unique episodes across the same 50-task split, with no missing or duplicate keys. Its strict summary is `runs/gdsq_week1_preregistered_v1/execution/aggregate/gr00t_uniform_w6/summary.json`, SHA256 `3a9165fe10fd69de8c23b254527135c4b30a5d20b8b7bf1030609e887ec75c36`. A separate three-manifest replay audit preserves the original launch hashes, records later source drift, validates every selector-disabled row, and reproduces the summary byte-for-byte.

## $\Omega$-QVLA Table 1 audit

- Paper source: `https://arxiv.org/pdf/2605.28803v1`; source record `runs/gdsq_extension_preregistered_v1/omega_qvla_table1/source_record.json`, SHA256 `9a94dbae0b053b32ceac08d7a2f6ba2156764d2180295d370b743b7d34297d8f`.
- Official code: `https://github.com/UCMP13753/Omega-QVLA`, pinned at commit `3727e2203568db43fc5fba06ee8686c1b47c044f`.
- Official packs: GR00T revision `01252878485fa9295b53ab833eb13d564f6911ad`; $\pi_{0.5}$ revision `f3b20e335ca8a03129ab8fe251385c042a13cfe3`.
- Five-configuration Table 2 preregistration: `runs/gdsq_extension_preregistered_v1/libero_table2_five_config_v2.json`, SHA256 `1f2bc757bc0adbefa6466a89bfcb24d6b6595628e0cbe0f147fa0cd3af23bf73`; the immutable four-configuration predecessor is retained for audit history.
- Reported protocol: four LIBERO suites, ten tasks per suite, ten held-out trials per task, and ten calibration trajectories per suite. The local reproduction fixes initialization offset 10 and eight denoising steps following the pinned release command.
- Table arithmetic: the GR00T FP16 displayed suite values average to 86.5, whereas the v1 table reports 87.0. The paper's 87.0 is preserved as a source transcription and the mismatch is exposed in the audit JSON.
- Release drift: v1 states GPTQ damping 0.01 while the pinned repository builder documentation uses 0.05; the released $\pi_{0.5}$ pack has ten per-step buckets while evaluation uses eight denoising steps. We load released packs unchanged and forbid rebuild/retuning from test results.
- The RoboCasa365 GR00T Atomic pack is recalibrated without test-result feedback. Its formal result has exactly 900 unique episodes (18 tasks, 50 seeds), 541 successes, zero formal failures, and 60.1\% task-macro SR with a 10,000-draw task-cluster interval of $[50.7,68.7]$. The strict summary SHA256 is `bd417d70306bca83a022bca5c5fd2e7cac740738bc1f05a95816acc3ed43916a`; its frozen manifest SHA256 is `f0181a0973521489f665a69de371dfaf2adffaee8598a5c4d3ed11c3c2abe41e`.
- The independent storage audit SHA256 is `cbd299f7dd28033a194fde6fcf94be7255b96ed094d267fd3252a1ace84a84a2`. It gives 0.599 GiB and 3.33$\times$ in the same theoretical tightly packed Linear scope as the main table, including W4 weights, GPTQ scales, FP16 block rotations, DiT permutation indices, and A4 scale tables. The physical 6.41-GiB evaluation pack stores dequantized FP16 tensors and dense rotations and is explicitly excluded from the packed-deployment claim.
- Table 1 enables only the complete GR00T Atomic $\Omega$-QVLA cell and its independently audited storage cells; all Composite and $\pi_{0.5}$ cells remain claim-disabled. Table 2 (`tables/omega_qvla_libero.tex`) contains only local FP16, QuantVLA W4A8, Uniform W6, $\Omega$-QVLA W4A4, and GDSQ-VLA rows; no author-reported values are shown, and all cells remain disabled until exact 4,000-episode joint coverage and SHA checks pass.

## Rejected or corrected records

- Removed `Improving Language Model Distillation Through Hidden State Matching`: the exact record could not be corroborated in arXiv, Crossref, OpenAlex, or DBLP during the audit.
- Corrected GPTQ from an arXiv-only 2022 record to ICLR 2023 while retaining arXiv `2210.17323`.
- Corrected ViDiT-Q to ICLR 2025 and Q-DiT to the complete eight-author CVPR 2025 record.
- Corrected GR00T N1 to retain the NVIDIA group author and added canonical arXiv identifiers to RoboCasa365 and QuantVLA.

## Build-level reference checks

- `main.bib` contains 43 entries.
- The compiled bibliography contains 43 cited entries; there are no uncited placeholders.
- BibTeX reports zero warnings, and the final LaTeX log has no undefined citation or cross-reference warning.
