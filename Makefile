PY ?= python3

.PHONY: selftest demo paper clean

selftest:
	$(PY) examples/run_selftests.py

demo:
	$(PY) examples/demo_w4a8.py

paper:
	$(MAKE) -C paper pdf

clean:
	$(MAKE) -C paper clean
