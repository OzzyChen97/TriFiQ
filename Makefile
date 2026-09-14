PY ?= python3

.PHONY: selftest demo

selftest:
	$(PY) examples/run_selftests.py

demo:
	$(PY) examples/demo_w4a8.py