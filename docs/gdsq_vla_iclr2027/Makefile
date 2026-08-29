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
	$(ROBOCASA_PY) $(REPO_ROOT)/scripts/tools/gdsq_evidence_registry.py
	$(ROBOCASA_PY) $(REPO_ROOT)/scripts/tools/render_gdsq_main_table.py --check
	$(ROBOCASA_PY) $(REPO_ROOT)/scripts/tools/render_gdsq_claim_status.py --check
	$(ROBOCASA_PY) $(REPO_ROOT)/scripts/tools/render_gdsq_component_ablation.py --check
	$(ROBOCASA_PY) $(REPO_ROOT)/scripts/tools/render_omega_qvla_libero_table.py --check
	$(ROBOCASA_PY) $(REPO_ROOT)/scripts/tools/check_gdsq_paper.py \
		--paper-dir $(CURDIR) \
		--report $(REPO_ROOT)/runs/gdsq_week1_preregistered_v1/audit/paper_gate.json

clean:
	rm -rf $(BUILD) $(PAPER).pdf
