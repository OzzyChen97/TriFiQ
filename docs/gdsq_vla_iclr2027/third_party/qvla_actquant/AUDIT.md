# QVLA / ActQuant RoboCasa365 reproduction audit

This directory freezes the papers used for the protocol-matched Table 1
reproduction. Machine-readable source, license, build, patch, and checkpoint
hashes are written by `scripts/tools/qvla_actquant_provenance.py` to
`runs/qvla_actquant_table1/provenance.json`. Its current file SHA256 is
`0d69032dc604ff6c3d93ffd0b7f9b6257ad8f965ddb12cd8e8b12681b1b1878e`.

## Frozen upstream inputs

- QVLA paper: arXiv:2602.03782v1, SHA256
  `e53902d1c8f1c07b555a18c058d64a62d415e6e1c0b581b41fe1f25007af17c4`.
  The clean checkout is commit
  `26cc4821a3be4c003d09d3c7997b38db2a347982`; its deterministic Git archive
  SHA256 is
  `326380daf07c8ecce52a6696c7085229bb23835c8c031f37dd530e88b7f954eb`.
  The checkout contains MIT-licensed OpenVLA/OpenVLA-OFT components and an
  Apache-2.0 UniVLA component; exact license hashes are in the provenance JSON.
- ActQuant paper: arXiv:2605.24011v3, SHA256
  `8a3371d2668a259508184e4064197ea5c7c0dd2228725ffade028e7c739200b0`.
  The clean checkout is commit
  `b64791125070652fe6b554e244fe809c79ef5246`; its deterministic Git archive
  SHA256 is
  `e5ee1117eb61e2e6c2e622ca859552511c39f14768395871c4a8c0c4eea0807d`.
  The repository is MIT licensed.
- The only upstream-tree modification is the replayable patch
  `patches/qvla_actquant/actquant_component_quantizer.patch`, SHA256
  `b233836bfa5797dc7581592373c1752175c007eb425a3171100bce137ca0daa9`.
  It adds exact mixed tensor-type manifests and the released ggml IQ/K
  codebooks to the graph-free component quantizer. A clean-clone apply check
  and byte-for-byte diff comparison pass.

## Paper, released-code, and local-adapter boundary

QVLA's paper and released implementation target OpenVLA-family backbones on
LIBERO. The local adapter preserves the released Hessian proxy, channel gate
menu `{0,2,4,8,16}`, greedy allocator, and row-wise symmetric fake-quant
semantics, while replacing target-name routing with Eagle/Qwen for GR00T N1.5
and SigLIP/PaliGemma for pi0.5. Projectors, output heads, and action modules are
excluded. The released implementation materializes fake-quantized full-precision
weights; the local adapter additionally writes real mixed-row codes, FP32
scales, offsets, and native-precision rows. This pack is decoded once to the
existing PyTorch policy, so no runtime-memory or latency claim is permitted.
QVLA calibration jobs are fail-closed to explicit, method-scoped uniform BF16
model loads and BF16 activation compute; every Linear/Conv weight must attest
BF16 before a proxy shard can be written. Ordinary pi0.5 callers retain
OpenPI's historical FP32 numerical islands; the uniform conversion is enabled
only by the QVLA calibration or QVLA formal-runtime environment.

ActQuant's paper evaluates OpenVLA-OFT and pi0.5 on LIBERO. Its paper says that
RBF kernels are used throughout, whereas the released pi0.5 HSIC script defaults
to an RBF hidden kernel and a linear action kernel. The local reproduction uses
that released-code default and records it in every HSIC shard. The paper uses
`alpha=1` for action-only pi0.5 Fisher and `alpha=1/2` as the mixed-pathway
default; this frozen RoboCasa365 protocol deliberately requires `alpha=1` for
both local continuous-action adapters. Fisher therefore uses the native
flow-matching loss and per-weight `grad^2`, with no categorical loss.

For pi0.5, the final artifact follows the released unified and standalone GGUF
export, `llama-quantize --imatrix` with exact per-tensor overrides, and released
LLM merge. A local deterministic SigLIP component merge is then required because
the public exporter has no mixed per-tensor vision allocation path. For GR00T,
which has no public ActQuant exporter or ggml execution graph, the local adapter
uses deterministic Eagle/Qwen names and the patched graph-free component
quantizer. Both paths dequantize final GGUF tensors into the existing PyTorch
services and make no C++ runtime claim.
ActQuant calibration, allocation, packing, and formal-service jobs are
fail-closed to explicit FP16 model loads. Every Linear/Conv weight must attest
FP16, HSIC and flow-loss activation compute is FP16, and these facts are bound
into each shard, allocation, pack manifest, and scheduler job hash. Numerically
sensitive one-dimensional normalization parameters may remain FP32 exactly as
in the official pi0.5 GGUF exporter; the protected action expert, action head,
and multimodal projector GEMMs remain FP16.
GPU job manifests also bind the model-family interpreter: GR00T jobs use the
`groot_test` environment, while pi0.5 jobs use the OpenPI environment that
provides its JAX/PyTorch loader dependencies. The interpreter path is part of
the scheduler job SHA.

## Calibration deviation

The official RoboCasa human-demonstration URL returned HTTP 404 at
`2026-08-31T08:14:14Z`. The preregistered fallback is therefore active:
`calibration_source=fp16_teacher_proxy` and
`source_protocol_equivalent=false`. Independent environment seeds start at
1000, failed as well as successful trajectories are retained, and formal seeds
0--49 are rejected by the freeze validator. This is a local proxy calibration,
not a claim to have reproduced either paper's LIBERO source protocol.

The first GR00T teacher launch resolved to BF16, and a second nominal-FP16
launch retained BF16 weights in nested Linear/Conv modules. Neither set is
eligible calibration evidence. Their files were preserved without deletion in
`runs/qvla_actquant_table1/calibration/`, with byte counts, journal hashes,
process identities, and rejection reasons recorded under
`runs/qvla_actquant_table1/quarantine/`. The replacement GR00T loader applies
the requested dtype to the complete loaded module, not only the outer
`from_pretrained` call. Before accepting any episode, the collector checks the
server's stable metadata identity against
`runs/qvla_actquant_table1/teacher_precision_attestations.json`; the freeze
validator independently requires `resolved=float16` and
`strict_all_linear_conv_fp16=true` for that exact server identity. Strict
GR00T calibration is written to the separate
`runs/qvla_actquant_table1/calibration_fp16/` tree. This fail-closed boundary
prevents the quarantined BF16 or mixed-precision rows from entering either the
512-episode QVLA set or its fixed 60-episode ActQuant subset.

Teacher collection initially uses non-evicting scale-out: 17 strict GR00T
services serve 24 disjoint hash-sharded workers (6/5/6 services for
atomic-seen, composite-seen, and composite-unseen). After all 512 pi0.5 proxy
episodes have been committed with no extra key, a fail-closed rebalancer
verifies and stops only the seven recorded pi0.5 teacher PIDs, then grows the
GR00T pool to 24 services (8/8/8). A completed GR00T shard is not relaunched.
Every placement is admitted from a live `nvidia-smi` free-memory snapshot with
a 4 GiB model-placement reserve; no unrelated PID is stopped. Reconnection
preserves all committed archives and journals. Exact pre/post GPU snapshots,
process identities, ports, runtime precision inventories, and the rebalancer
state are recorded in `runs/qvla_actquant_table1/unattended/`.
The pi0.5 tail was likewise resharded only after all live collector identities
were checked: 19 disjoint hash shards feed the seven unchanged FP16 teacher
services, while stale PID files are recorded and never signalled. This changes
only rollout concurrency; episode selection, environment seeds, teacher
identity, and committed archives remain frozen.
A locked watchdog checks only missing-shard ownership. If a required collector
dies, it invokes the same identity-checking reshard/scale script; it neither
signals a healthy worker nor performs GPU eviction. Recovery events and stale
PID observations are persisted beside the controller state.

No success-rate result may enter the paper from this pipeline until the pack
manifest is frozen, both smoke gates pass, and the strict aggregator observes
exactly 2,500 unique successful-or-failed environment completions with zero
infrastructure failures for each model-method row.
