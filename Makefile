# probe — build, experiment and evaluation entry points.
#
# Every number in the README is produced by one of these targets. Nothing is
# hand-typed. `make eval` is the reproducibility contract: run it twice on the
# same traces and the results table is byte-identical.

UV       ?= uv
PY       := $(UV) run python
PYTEST   := $(UV) run pytest
PROBE    := $(UV) run probe

# Backend for LLM roles: sim (default, deterministic offline) | fake | anthropic
BACKEND  ?= sim
SEED     ?= 20260801
TRACES   ?= traces/probe.duckdb

.DEFAULT_GOAL := help

.PHONY: help sync test lint fmt clean \
        taxonomy bank population calibrate experiment eval figures report demo viewer \
        gate-0 gate-1 gate-2 gate-3 gate-4 gate-5 gate-6 gates

help:  ## list every target
	@$(PY) scripts/mk_help.py

sync:  ## install the project and dev dependencies
	$(UV) sync --all-extras

test:  ## full unit + property + statistical suite
	$(PYTEST)

lint:
	$(UV) run ruff check probe evals tests

fmt:
	$(UV) run ruff format probe evals tests

# ---------------------------------------------------------------- pipeline --

taxonomy:  ## validate the competency taxonomy
	$(PROBE) taxonomy validate

bank:  ## generate the raw (uncalibrated) question bank
	$(PROBE) bank generate --seed $(SEED)

population:  ## generate the persona population with hidden theta*
	$(PROBE) population generate --seed $(SEED)

calibrate:  ## fit GRM item parameters on the calibration split, emit bank vN
	$(PROBE) bank calibrate --seed $(SEED)

experiment:  ## re-run every interview from scratch (prints a cost estimate first)
	$(PROBE) experiment run --backend $(BACKEND) --seed $(SEED) --traces $(TRACES)

eval:  ## compute every metric from committed traces -> results table + figures
	$(PY) -m evals.run_eval --traces $(TRACES) --suites evals/suites

figures:
	$(PY) -m evals.run_eval --traces $(TRACES) --suites evals/suites --figures-only

report:  ## regenerate README artefact blocks and the 2-page PDF
	$(PY) -m analysis.build_report

demo:  ## render the side-by-side fixed-vs-eig adversarial demo
	$(PROBE) demo render --traces $(TRACES)

viewer:  ## render one run: transcript + belief trajectory
	$(PROBE) viewer show --traces $(TRACES) --run-id $(RUN)

# ------------------------------------------------------------- phase gates --

gate-0:
	$(PYTEST) tests/phase0 -m "not slow"

gate-1:
	$(PYTEST) tests/phase0 tests/phase1

gate-2:
	$(PYTEST) tests/phase0 tests/phase1 tests/phase2

gate-3:
	$(PYTEST) tests/phase0 tests/phase1 tests/phase2 tests/phase3

gate-4:
	$(PYTEST) tests/phase0 tests/phase1 tests/phase2 tests/phase3 tests/phase4

gate-5:
	$(PYTEST) tests/phase0 tests/phase1 tests/phase2 tests/phase3 tests/phase4 tests/phase5

gate-6:
	$(PYTEST) --cov=probe.belief --cov=probe.policy --cov=evals.metrics --cov-report=term-missing --cov-fail-under=90

gates: gate-6

clean:  ## remove caches and scratch traces
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.hypothesis','htmlcov','traces/scratch']]"
