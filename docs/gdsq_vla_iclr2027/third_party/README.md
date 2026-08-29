# Paper-local TeX compatibility files

The host uses TeX Live 2017, while the paper source uses several newer package
features. This directory vendors the generated `.sty` files for:

- `silence` from CTAN (`macros/latex/contrib/silence`), LPPL;
- `enumitem` from CTAN (`macros/latex/contrib/enumitem`), MIT;
- `cleveref` from CTAN (`macros/latex/contrib/cleveref`), LPPL.

They are included only to make the unchanged official ICLR 2027 source
reproducible on this machine. Distribution remains governed by each package's
upstream license.
