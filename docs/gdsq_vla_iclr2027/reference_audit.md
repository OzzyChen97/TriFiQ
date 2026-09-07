# Reference audit

> Result-policy note (2026-08-30): citation metadata below remains active, while the historical
> GDSQ result-policy sections are retained as provenance. Current DyPAC-VLA headline claims and
> hashes are authoritative in `dypac_evidence_registry.json` and `FINAL_VERSIONS.md`.

Audit date: 2026-09-03. This file is internal and is not included in the submission PDF.

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
- Adjacent VLA compression: QVLA `2602.03782`, ActQuant `2605.24011v3`, $\Omega$-QVLA `2605.28803v1`, and SQAP-VLA `2509.09090`.

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

`tables/main_results.tex` is the unified RoboCasa365 comparison. It groups Atomic, Composite-Seen, and Composite-Unseen columns with their task-weighted 50-task macro Mean, then groups rows by GR00T N1.5 and $\pi_{0.5}$. Final task-set cells require exact task/seed coverage and an audited summary. During an active registered run, a content-bound partial snapshot may be inserted only with italic values, an explicit interim marker, exact per-arm coverage, no paired inference, and a statement that uneven coverage is descriptive rather than a final comparison; values from different protocols are never combined.

- The $\pi_{0.5}$ Uniform-W6 control is complete at exactly 2,500 unique episodes (900 Atomic, 800 Composite-Seen, and 800 Composite-Unseen), with no missing or duplicate keys. Its strict summary is `runs/gdsq_week1_preregistered_v1/execution/runs/pi05_uniform_w6_official50/aggregate/summary.json`, SHA256 `849e1e159daf6b515bcb1724a8862fea17e9c0c95053ed5fefc99b9c145e052a`; the frozen manifest SHA256 is `7abe10e3389a362dc118a9f7c50da542f72bdf15d7517f66aa2c8fb3c1cce476`.
- The GR00T Uniform-W6 control is complete at exactly 2,500 unique episodes across the same 50-task split, with no missing or duplicate keys. Its strict summary is `runs/gdsq_week1_preregistered_v1/execution/aggregate/gr00t_uniform_w6/summary.json`, SHA256 `3a9165fe10fd69de8c23b254527135c4b30a5d20b8b7bf1030609e887ec75c36`. A separate three-manifest replay audit preserves the original launch hashes, records later source drift, validates every selector-disabled row, and reproduces the summary byte-for-byte.
- The clean-room DA-PTQ cross-architecture adaptation is complete for both models at 2,500/2,500 episodes each. GR00T records 76 successes (Atomic/Composite-Seen/Composite-Unseen: 73/0/3), giving 3.0%; $\pi_{0.5}$ records 655 successes (527/98/30), giving 26.2%. The strict aggregate is `runs/daptq_table1/formal/aggregate.json`, SHA256 `2c857eab542035650a22bfd54a890fed49afd084e999ee5f17c37adcddf9eeb8`; it validates all 500 receipts, every output and artifact hash, four-flow-step metadata, and no test-result feedback. The frozen protocol marks `source_protocol_equivalent=false`. Against the paired DyPAC-VLA rows, DA-PTQ has 8/1,282 wins/losses on GR00T and 182/220 on $\pi_{0.5}$ (Holm $p=0.0648466$ for the latter).
- The post-selection GR00T FCP diagnostic is complete for all six preregistered proposals at 500 episodes each (50 tasks, seeds 0--9), totaling 3,000 new rollouts. Its aggregate is `runs/full_context_v2/fcp_completion/closed_loop/aggregate.json`, SHA256 `1cedce9f204bd3f265cbf1d604020253c78f89791655b914599582d8891f3c77`; the preregistration SHA256 is `d7048cabe5b82c1cb7ce322df911c2c96b278f369bfb6448ea487dc6a0c18bd6`. Candidate success spans 50.6--56.2% versus 53.8% for the paired frozen anchor. None differs under the fixed six-test Holm family (minimum adjusted $p=0.891$). Outcome labels were unavailable to the frozen decision, the selected mask is unchanged, and the null findings are not interpreted as equivalence.
- The preregistered $\pi_{0.5}$ same-protocol FCP diagnostic is complete at 500 paired task--seed keys per configuration. It adds 1,500 episodes for the transferred initializer and two post-projection proposals, and reuses 500 frozen-anchor episodes. The raw aggregate and summary SHA256 values are `7fb09f82d1fbe51ba4ee1af0f1a4d9671f2cce9255882951609242247630601c` and `6fdcdf4c53f86ff45cf00cee4fe50abbc9c3ad8ceeab5005dbee531d1f19e87e`. The projected anchor records 28.2% versus 26.0% for the over-budget initializer ($+2.2$ points; exact McNemar $p=0.2351$). Single-best and two-best record 24.8% and 25.0%; both have Holm-adjusted $p=0.0859$ against the anchor. The diagnostic supports budget adaptation without an observed drop in this sample, but establishes neither a gain nor non-inferiority and cannot revise the frozen choice.
- The formal-scale maximal-rate rows are complete at 2,500/2,500 episodes each (50 tasks, seeds 0--49, shared four-flow-step protocol). GR00T (all-W4 rate-family endpoint) reaches 52.8% task-macro success (1319/2500; 70.9/42.9/42.3 by split) at 836,960,256 static bytes (2.556$\times$); $\pi_{0.5}$ (180-layer all-W4 profile) reaches 26.8% (670/2500; 58.7/14.5/3.3 by split) at 1,242,169,344 static bytes (3.556$\times$). The combined aggregate is `runs/robocasa365_table1_max_sweep_v1/aggregate.json`, SHA256 `93bb7b021ca83b073b314c6b18e97f3d7b4c4e327eeb190375d7d5fedb3070bc`; the plans are `runs/robocasa365_table1_max_sweep_v1/masks/gr00t_max.plan.json` (SHA256 `b1fe584404184bf6f5505af5ef40ebaeab759a94c1e0e952dd19b085e387179d`) and `runs/robocasa365_table1_max_sweep_v1/masks/pi05_max.plan.json` (SHA256 `8b0aabc15e48970250e605ca689e08bde3c9b73616d2c31fd2fae88d01759dbd`). Cross-run pairing against the audited anchors (54.0% GR00T; 27.7% $\pi_{0.5}$) gives 319/288 (exact McNemar $p=0.223$) and 182/159 ($p=0.233$) discordant pairs. Both rows are displayed as \method (Ours, max rate) with an explicit annotation; they are descriptive operating points, not registered tests, and neither revises the frozen headline rows.
- A post-hoc real-integer W4A8 deployment audit is reported only in the hardware appendix. Its complete paired 500-episode rerun and separate A40 A/B are bound by `evidence/integer_w4a8_backend_audit.json`, SHA256 `8e12ed20550dff16f999af2383a5ae89c49e2d48f17b1deb04ad9a104447c219`; the source aggregate and hardware hashes are `35790611590b135f6b3ff41f69561e81aaec04d5b1ccb5b912138eefc7410498` and `50cc2f2899224f39af9575851b2523f4ab182ffc6274f0a4f1ea5a18636bd033`. Because kernel compatibility changes dynamic-A8 scaling from per-input-channel to per-row group-64, this point neither reopens FCP nor replaces the formal Ours row. The appendix hardware table intentionally contains no success-rate or delta-success columns.

## Runtime-reporting audit

- QuantVLA v4 reports task success and memory reduction but no measured latency or energy.
- QVLA reports success, memory, and relative speedup in its primary trade-off table.
- DA-PTQ reports memory reduction and relative speedup for its source setup.
- ActQuant v3 provides a dedicated inference-speed section with absolute per-token latency on A6000, Apple M4, and Jetson Thor.
- HoloQ-VLA (the later revision of $\Omega$-QVLA) includes an appendix real-INT4 kernel benchmark for complete `sample_actions`, reporting batch-one and batch-eight speedups.

These precedents support keeping runtime evidence separate from policy-quality tables. Our submission therefore places absolute A40 latency, energy, and peak allocation in the appendix and does not mix backend success-rate columns into the main FCP table.

## $\Omega$-QVLA Table 1 audit

- Paper source: `https://arxiv.org/pdf/2605.28803v1`; source record `runs/gdsq_extension_preregistered_v1/omega_qvla_table1/source_record.json`, SHA256 `9a94dbae0b053b32ceac08d7a2f6ba2156764d2180295d370b743b7d34297d8f`.
- Official code: `https://github.com/UCMP13753/Omega-QVLA`, pinned at commit `3727e2203568db43fc5fba06ee8686c1b47c044f`.
- Official packs: GR00T revision `01252878485fa9295b53ab833eb13d564f6911ad`; $\pi_{0.5}$ revision `f3b20e335ca8a03129ab8fe251385c042a13cfe3`.
- Five-configuration Table 2 preregistration: `runs/gdsq_extension_preregistered_v1/libero_table2_five_config_v2.json`, SHA256 `1f2bc757bc0adbefa6466a89bfcb24d6b6595628e0cbe0f147fa0cd3af23bf73`; the immutable four-configuration predecessor is retained for audit history.
- Reported protocol: four LIBERO suites, ten tasks per suite, ten held-out trials per task, and ten calibration trajectories per suite. The local reproduction fixes initialization offset 10 and eight denoising steps following the pinned release command.
- Table arithmetic: the GR00T FP16 displayed suite values average to 86.5, whereas the v1 table reports 87.0. The paper's 87.0 is preserved as a source transcription and the mismatch is exposed in the audit JSON.
- Release drift: v1 states GPTQ damping 0.01 while the pinned repository builder documentation uses 0.05; the released $\pi_{0.5}$ pack has ten per-step buckets while evaluation uses eight denoising steps. We load released packs unchanged and forbid rebuild/retuning from test results.
- The three RoboCasa365 GR00T packs are recalibrated without test-result feedback. Atomic has 900 unique episodes, 541 successes, and 60.1\% SR; Composite-Seen has 800 episodes, 207 successes, and 25.9\% SR; Composite-Unseen has 800 episodes, 220 successes, and 27.5\% SR. Every task has exactly 50 seeds and every subset has zero formal failures. The summary/manifest SHA256 pairs are Atomic `bd417d70306bca83a022bca5c5fd2e7cac740738bc1f05a95816acc3ed43916a` / `f0181a0973521489f665a69de371dfaf2adffaee8598a5c4d3ed11c3c2abe41e`, Composite-Seen `8b1f229d7ad78b6f1a3fa1a2010a3a760e220f6d799b6db723af7a800a8be857` / `71da3382b7be0a9857958bf2230a5c1c7a3c898a293d48a9cc9455fd409c7997`, and Composite-Unseen `693bbfecdcaebd54475899e90cc35346fa2e2e26477c4c7f1ff01c190cb7e233` / `4f81a12ab8ed7152551a25acbc75e59352e2e887fe2c1d0a6a0b39255821b040`.
- The strict 50-task aggregate contains 2,500 episodes and 968 successes, giving 38.7\% task-macro SR; its SHA256 is `88f5a0ffd282148ed98f5c6df3d2087c5b49a76dda5ff4f84e4f14a71b8e749b`. The final eight Composite-Unseen rows replace only infrastructure-timeout gaps under the unchanged parent manifest. The repair audit SHA256 `926af56182e8de7fc1d433a548235ffcd92a8793589327d7cb45ebb8f6b667c6` verifies frozen runtime hashes, the pinned external commit, exact 50-row repaired-shard coverage, and no remaining missing key; three inactive D-PAC-only source paths are explicitly disclosed and excluded from the $\Omega$-QVLA runtime.
- The independent storage audit SHA256 is `cbd299f7dd28033a194fde6fcf94be7255b96ed094d267fd3252a1ace84a84a2`. It gives 0.599 GiB and 3.33$\times$ in the same theoretical tightly packed Linear scope as the main table, including W4 weights, GPTQ scales, FP16 block rotations, DiT permutation indices, and A4 scale tables. The physical 6.41-GiB evaluation pack stores dequantized FP16 tensors and dense rotations and is explicitly excluded from the packed-deployment claim.
- The independent $\pi_{0.5}$ $\Omega$-QVLA reproduction is complete at 2,500 unique episodes. Atomic has 900 episodes, 446 successes, and 49.6\% task-macro SR; its summary/manifest SHA256 pair is `40ea63fff04bfdbfb3faa13895232d498f70146fd2ce858b5a63c06cfba68de9` / `9926d70abdaaa57b063b391a96a00b9f2d5a302c531632ff35fcf18b3a068c7d`. Composite-Seen has 800 episodes, 83 successes, and 10.4\% SR; its pair is `854854206880c83237c76cb51d195dd2b48aac046e5457b7721d60d99fbea8c0` / `d3604ed2a076b4ea88e978126b406374d07b1c3718f621567d6587603ebda08e`. Composite-Unseen has 800 episodes, 8 successes, and 1.0\% SR; its pair is `d567c8d672b5dc57e122b0a6ba73633c3dfdbd9c192e45503e370d974034e571` / `b52c670fd5b4e0974ef964c669417dfbeb76868f8cc24c03a793289216d1e34b`. The strict 50-task aggregate gives 21.5\% with SHA256 `d898552b26fb68db253decfe9375977f1dbf38daf89bb94d73c6d62564114ae9`. All subsets have zero missing, duplicate, or failed episodes.
- The formal $\pi_{0.5}$ DyPAC-VLA evaluation uses the same four-flow-step protocol as the completed references and is complete at 2,500 unique episodes with zero missing, duplicate, conflicting, or failed rows. Atomic records 536/900 successes (59.6\%), Composite-Seen records 123/800 (15.4\%), and Composite-Unseen records 34/800 (4.3\%), for 693/2,500 overall (27.7\%). The strict aggregate is `runs/full_context_v2/pi05_table1/aggregate.json`, SHA256 `512480a2e0836423254217b5215489bcb218e17a4c7d0fff1cdf5aba9acf4f73`; correcting the previously mislabeled ten-step metadata changes no episode outcome or coverage key. The correction audit is `runs/full_context_v2/pi05_table1/protocol_correction.json`, SHA256 `cd5e07baaaf7b40e53c1ace9a881eb49aabb9407d8d20189d46e3a11b2186770`.
- The $\pi_{0.5}$ storage audit SHA256 is `38e8af86473a566a23bcb52801a5e951a576135d9f905d19db907c067adf82f6`. It verifies all 252 PaliGemma/expert W4A4 records and gives 1.307 GiB, or 3.27$\times$ relative to the corresponding 4.271-GiB FP16 scope. The 34.97-GiB evaluation pack is dequantized and is not reported as deployment storage.
- Table 1 enables complete 50-task $\Omega$-QVLA and ActQuant rows with independently audited storage cells; QVLA-code is omitted and its failed cross-architecture reproduction is disclosed in the appendix. Main-paper Table 2 (`tables/omega_qvla_libero.tex`) reports the complete LIBERO suite breakdown, model-level averages, and static size; the compact generated view is not referenced. Its joint gate covers 4,000/4,000 episodes, 40/40 complete cells, and no missing or duplicate keys. The GR00T values for FP16, QuantVLA, Uniform W6, $\Omega$-QVLA, and ours are 1.993/0.898/1.109/0.599/0.948 GiB. The corresponding $\pi_{0.5}$ values are 4.113/1.388/1.902/1.307/1.788 GiB. Suite-specific masks use the arithmetic mean of the four exact plan sizes. The two separately marked 4.0-BPW $\pi_{0.5}$ anchors from ActQuant v3 Table 1 report QVLA at Goal/Spatial/Object/Long/Avg. = 96.4/98.0/97.2/91.8/95.8 and ActQuant at 96.8/98.4/99.4/91.8/96.6, with Mem. 2.7 GB for both. These two rows do not enter the local 4,000-episode denominator, and their Mem. measure is not conflated with the locally audited static Size values. The supplied image SHA256 is `279b79fb5fc2b8cce6bc82db89cdd9cd1404289d73c52fde9bfcf137035179a5`.

## Rejected or corrected records

- Removed `Improving Language Model Distillation Through Hidden State Matching`: the exact record could not be corroborated in arXiv, Crossref, OpenAlex, or DBLP during the audit.
- Corrected GPTQ from an arXiv-only 2022 record to ICLR 2023 while retaining arXiv `2210.17323`.
- Corrected ViDiT-Q to ICLR 2025 and Q-DiT to the complete eight-author CVPR 2025 record.
- Corrected GR00T N1 to retain the NVIDIA group author and added canonical arXiv identifiers to RoboCasa365 and QuantVLA.

## Build-level reference checks

- `main.bib` contains 52 verified metadata records; it is a superset of the active citations.
- The compiled bibliography contains 43 cited entries (4.78 per main-paper page), exceeding the adapted three-citations-per-page final-paper gate. Nine verified background records remain uncited and do not enter the compiled bibliography.
- Of the 43 active citations, 21 are dated 2025--2026 (48.8\%), 28 have a proceedings or non-arXiv journal record (65.1\%), and 15 remain arXiv-only (34.9\%). These year-level counts pass the protocol's recency, accepted-work, and arXiv-ratio heuristics for this nine-page empirical paper.
- BibTeX reports zero warnings, and the final LaTeX log has no undefined citation or cross-reference warning.

<!-- BEGIN QVLA_ACTQUANT_TABLE1 -->
## ActQuant local reproduction and QVLA failure audit

- QVLA semantic source: arXiv:2602.03782v1 and official commit `26cc4821a3be4c003d09d3c7997b38db2a347982`.
- ActQuant semantic source: arXiv:2605.24011v3 and official commit `b64791125070652fe6b554e244fe809c79ef5246`.
- ActQuant follows the pinned official GitHub implementation for allocation and quantization semantics. Local code is limited to model-specific tensor routing, artifact packaging, calibration integration, and RoboCasa365 evaluation.
- The exact frozen `scripts/inference_service.py` used by the rollout is retained in the content-addressed `runs/qvla_actquant_table1/frozen_sources/` archive; finalization verifies it against the pre-rollout implementation SHA rather than the later working-tree revision.
- QVLA-code followed the pinned official GitHub quantizer and channel-allocation implementation, but the public release contains no GR00T, pi0.5, RoboCasa365, or corresponding mixed-row export path. The local adapters produced 0/1,728 GR00T and 0/1,517 pi0.5 successes before the run was stopped. The public artifacts cannot distinguish incomplete open sourcing from an error in the released code path; QVLA-code is omitted from Table 1 and this is not treated as a failure of the native OpenVLA/LIBERO setup.
- Calibration is labeled `fp16_teacher_proxy` and `source_protocol_equivalent=false`; results are reported as measured without post-hoc tuning.
- Every reported Table-1 candidate has exactly 2,500 unique target-split episodes and a frozen pack/runtime metadata chain. Runtime claims are restricted to the separate A40 GR00T measurement protocol; no ActQuant latency or live-memory saving is inferred from static pack size.

- `actquant_gr00t_vs_dypac_gr00t`: wins/losses 340/368; exact McNemar p=0.310237, Holm p=0.310237.
- `actquant_pi05_vs_dypac_pi05`: wins/losses 135/199; exact McNemar p=0.000546856, Holm p=0.00109371.
<!-- END QVLA_ACTQUANT_TABLE1 -->
