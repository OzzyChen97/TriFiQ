# Full-context FP16 protection v1 results

This directory contains the compact, reviewable evidence retained from the
`codex/d-pac-softfold` experiments. The branch intentionally does not contain
multi-gigabyte checkpoints, FP16 captures, Hessian NPZ payloads, identity
packs, process-control files, or per-GPU score shards.

Retained evidence includes:

- the final cross-model audit and source hashes;
- paired quick-evaluation rows and frozen quick aggregates;
- merged per-layer counterfactual scores and their candidate manifests;
- full-network proposal scores, frozen plans, and selection reports;
- real-W4 runtime metadata showing packed residency and zero FP-sized W4
  buffers.

The omitted binary artifacts remain identified by SHA-256 in the retained
metadata. They can be regenerated from the frozen checkpoint, protocol,
calibration buffer, and scripts in this branch.

Neither GR00T N1.5 nor π0.5 passed the preregistered quick advancement gate,
so Table 1 was not started. See `final_audit.json` for the terminal summary.
