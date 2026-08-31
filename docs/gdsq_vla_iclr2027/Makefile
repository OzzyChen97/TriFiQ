PAPER := main
BUILD := .build
TEXMF := $(CURDIR)/third_party/texmf
REPO_ROOT := $(abspath ../..)
ROBOCASA_PY := /home1/gyy/probe/miniforge3/envs/robocasa365/bin/python

export TEXINPUTS := $(TEXMF)//:
export BIBINPUTS := $(CURDIR):
export BSTINPUTS := $(CURDIR):

.PHONY: pdf check clean

pdf:
	@mkdir -p $(BUILD)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(PAPER).tex
	cd $(BUILD) && bibtex $(PAPER)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(PAPER).tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(PAPER).tex
	cp $(BUILD)/$(PAPER).pdf $(PAPER).pdf

check: pdf
	$(ROBOCASA_PY) $(REPO_ROOT)/scripts/tools/render_omega_qvla_libero_table.py --check
	$(ROBOCASA_PY) $(REPO_ROOT)/scripts/tools/render_dypac_paper.py --check
	$(ROBOCASA_PY) $(REPO_ROOT)/scripts/tools/check_gdsq_paper.py \
		--paper-dir $(CURDIR) \
		--report $(REPO_ROOT)/runs/full_context_v2/paper_audit.json

clean:
	rm -rf $(BUILD) $(PAPER).pdf
