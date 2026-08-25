# GDSQ-VLA CVPR 2026 draft

This directory contains an anonymous CVPR 2026 review draft using the official
`CVPR2026-v1(latex)` author kit at commit
`12909ae437f6dbc7435069cfdb4ca44c18e6a02f`.

Build the paper with:

```bash
make pdf
```

The build intentionally uses a paper-local TeX tree for three packages missing
from the host's TeX Live 2017 installation. The official `cvpr.sty` is unchanged.

`figures/gdsq_pipeline.png` is the user-selected, title-free version of selector-aware
candidate D. The original image, prompts, alternative candidates, and provenance
sidecars are retained under `figures/candidates/` and `figures/prompts/`; no API
credential is copied into this directory. Result tables are generated from audited
records, and pending cells may be replaced only by complete frozen summaries.
