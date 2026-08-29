# GDSQ-VLA ICLR 2027 draft

This directory contains an anonymous ICLR 2027 review draft using the official
author kit downloaded from the ICLR 2027 Author Guidelines. The downloaded ZIP
has SHA256 `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.
The bundled `iclr2027_conference.sty` and `iclr2027_conference.bst` are
unmodified and have SHA256 values
`797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6`
and `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5`,
respectively.

Build the paper with:

```bash
make pdf
```

The build intentionally uses a paper-local TeX tree for three packages missing
from the host's TeX Live 2017 installation. The ICLR style, bibliography style,
and its bundled `natbib.sty` and `fancyhdr.sty` are retained verbatim.

The anonymous submission gate enforces the ICLR 2027 nine-page main-text limit,
the required AI use statement, appendix placement after the bibliography,
official style hashes, readable booktabs tables, and a warning-free PDF.

`figures/gdsq_pipeline.png` is the user-selected, title-free version of selector-aware
candidate D. The original image, prompts, alternative candidates, and provenance
sidecars are retained under `figures/candidates/` and `figures/prompts/`; no API
credential is copied into this directory. Result tables are generated from audited
records, and pending cells may be replaced only by complete frozen summaries.
