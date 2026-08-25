.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	pytest -v --color=yes tests/

.PHONY : format
format :
	isort .
	black .

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build

ROBOCASA_PY := /home1/gyy/probe/miniforge3/envs/robocasa365/bin/python
GROOT_PY := /home1/gyy/probe/miniforge3/envs/groot_test/bin/python
OPENPI_PY := /home1/gyy/probe/miniforge3/envs/openpi/bin/python

.PHONY: test-robocasa test-gr00t test-openpi test-paper test-gdsq

test-robocasa:
	$(ROBOCASA_PY) -m pytest -q \
		scripts/tools/test_gdsq_evidence_registry.py \
		scripts/tools/test_render_gdsq_main_table.py \
		scripts/tools/test_render_gdsq_component_ablation.py \
		scripts/tools/test_pi05_selector_official_manifest.py \
		scripts/tools/test_pi05_week1_controls.py \
		scripts/tools/test_gdsq_gr00t_selector_reuse.py \
		scripts/tools/test_gdsq_finalize_week1.py \
		scripts/tools/test_gdsq_extension_plan.py \
		scripts/tools/test_omega_qvla_table1.py
	$(ROBOCASA_PY) scripts/tools/run_crit4_trial_driver.py \
		--port 1 --tasks SelfTest --selftest

test-gr00t:
	$(GROOT_PY) -m pytest -q \
		scripts/tools/test_gr00t_runtime_selector.py \
		scripts/tools/test_atmohb_dynamic_selector.py \
		scripts/tools/test_gdsq_week1_gr00t_screen.py \
		scripts/tools/test_download_omega_qvla_packs.py

test-openpi:
	cd code/pi05/openpi && PYTHONPATH=src $(OPENPI_PY) -m pytest -q \
		../../../scripts/tools/test_pi05_runtime_selector.py \
		../../../scripts/tools/test_pi05_paper_atm_ohb.py \
		../../../scripts/tools/test_gdsq_week1_pi05_screen.py \
		tools/test_pi05_formal_protocol.py

test-paper:
	$(MAKE) -C docs/gdsq_vla_cvpr2026 check

test-gdsq: test-robocasa test-gr00t test-openpi test-paper
