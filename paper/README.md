# Anonymous submission — paper source

LaTeX source, generated figures and tables, and the compiled PDF of the anonymous submission
**DyPAC-VLA: Representation-to-Action Fidelity for Low-Bit Vision-Language-Action Models**.

```bash
make pdf      # rebuild main.pdf with pdflatex + bibtex
```

`main.pdf` is the compiled submission. `main.tex` uses the unmodified official ICLR 2027 author
kit (`iclr2027_conference.sty`, `iclr2027_conference.bst`); `third_party/texmf/` vendors the
`enumitem` package so the source also builds on an older TeX Live.

## Layout

- `sections/`: abstract, introduction, related work, method, experiments, conclusion, statements,
  and supplement.
- `tables/`: generated result tables.
- `figures/`: active figure wrappers and their rendered assets.
- `dypac_evidence_registry.json`, `experiment_registry.json`, `tables/*.audit.json`,
  `evidence/*.json`: frozen provenance records for the reported numbers. They reference run
  directories from the full experimental pipeline, which this minimal release does not
  redistribute.

## Notes

- The submission is anonymous; `\iclrfinalcopy` stays commented out.
- Every table and figure in this directory is referenced by the manuscript source.
- The `check` target of the full research repository (which re-derives the tables from frozen run
  directories and enforces submission gates) is not part of this minimal release.
