PYTHON ?= python3
CODE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONPATH := $(CODE_DIR)src:$(PYTHONPATH)

.DEFAULT_GOAL := check
.PHONY: test check check-full scientific-check smoke study graphs analysis figures paper package reproduce

test:
	$(PYTHON) -m pytest "$(CODE_DIR)tests" -p no:cacheprovider

check:
	$(PYTHON) "$(CODE_DIR)scripts/check_release.py"

check-full:
	$(PYTHON) "$(CODE_DIR)scripts/check_release.py" --full

scientific-check:
	$(PYTHON) -m gcqaoa.verify

smoke:
	$(PYTHON) "$(CODE_DIR)scripts/run_smoke.py"

study:
	$(PYTHON) "$(CODE_DIR)scripts/run_study.py"

graphs:
	$(PYTHON) "$(CODE_DIR)scripts/generate_graphs.py"

analysis:
	$(PYTHON) "$(CODE_DIR)scripts/analyze_results.py"

figures:
	$(PYTHON) "$(CODE_DIR)scripts/make_figures.py"

paper: analysis figures
	$(PYTHON) "$(CODE_DIR)scripts/build_paper.py"

package:
	$(PYTHON) "$(CODE_DIR)scripts/package_arxiv.py"

reproduce:
	$(PYTHON) "$(CODE_DIR)scripts/generate_graphs.py"
	$(PYTHON) "$(CODE_DIR)scripts/analyze_results.py"
	$(PYTHON) "$(CODE_DIR)scripts/make_figures.py"
	$(PYTHON) "$(CODE_DIR)scripts/build_paper.py"
	$(PYTHON) "$(CODE_DIR)scripts/package_arxiv.py" --check
	$(PYTHON) "$(CODE_DIR)scripts/check_release.py" --write-manifest
