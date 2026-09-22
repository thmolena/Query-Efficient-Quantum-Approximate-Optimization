# Query-Efficient Quantum Approximate Optimization via Graph-Conditioned Trust Regions

Molena Huynh · North Carolina State University

[Paper](submission/main.pdf) · [Project page](index.html) · [Method](code/docs/METHOD.md) · [Evidence](code/docs/EVIDENCE.md)

Graph information can provide a useful starting point for the quantum approximate optimization algorithm (QAOA). Does local refinement then justify its measurement cost? This project separates **parameter transfer, search geometry, proposal quality, and statistical acceptance** in finite-shot MaxCut optimization. It includes the implementation, complete experiment records, independent numerical checks, and the source of every manuscript figure and table.

The central finding is a separation between initialization and refinement. In the canonical study, descriptor-selected anchors meet the stated target on all 24 test graphs at depths 1 and 2. Nine undisplaced donor-based local-model policies return those policies' respective anchors after spending **196,341,504 online shots across 2,592 runs**, with no accepted refinement. A separate follow-up records a few accepted variance-sensitive steps; SPSA and COBYLA achieve lower mean reference deficits at both follow-up depths. These are ideal-circuit simulations on small graphs, not a demonstrated measurement-complexity advantage.

## Quick start

Use Python **3.13**. From this folder:

```sh
python3.13 -m venv code/.venv
source code/.venv/bin/activate
python -m pip install -r code/requirements-lock.txt
python -m pip install --no-deps -e code
python code/scripts/run_smoke.py --config code/configs/smoke.json
```

The smoke example performs 56 optimizer runs and prints:

```text
Wrote 56 runs to .../code/results/executions/EXECUTION_ID/study.json.gz; ... seconds; EXECUTION_ID
```

The identifier and time vary. Smoke records are installation checks, separate from the paper evidence. New executions use fresh directories and preserve existing records. Their directory is ignored by Git and excluded from the supplied-file inventory.

A minimal circuit calculation:

```python
import networkx as nx
import numpy as np
from gcqaoa.qaoa import MaxCutQAOA, exact_maxcut

circuit = MaxCutQAOA(nx.cycle_graph(5))
theta = np.array([0.7, 0.25])  # gamma_1, beta_1
objective = circuit.objective(theta)
mean, variance_of_mean = circuit.sample(theta, 4096, np.random.default_rng(17))
maximum_cut, _ = exact_maxcut(nx.cycle_graph(5))
print(f"Expected approximation ratio: {-objective * 5 / maximum_cut:.6f}")
```

The objective is minus expected cut divided by the number of edges, so smaller is better. `sample` returns a mean and an unbiased estimate of its variance. Angles list all cost angles before all mixer angles. Graphs are simple, undirected, unweighted, and have no self-loops.

## Reproduce the paper

The supplied folder is self-contained: **no Git history, remote, branch, or tag is needed**. Exact source snapshots preserve the provenance of the recorded experiments even when the current analysis code has changed. See [source provenance](code/docs/PROVENANCE.md).

Run the tests and regenerate the summaries, figures, tables, and matching PDF:

```sh
python -m pytest code/tests -p no:cacheprovider -q
python code/scripts/analyze_results.py --config code/configs/paper.json
python code/scripts/make_figures.py --config code/configs/paper.json
python code/scripts/build_paper.py
```

The PDF builds from the authoritative source `submission/main.tex`. It needs **Tectonic** (which downloads its standard TeX bundle on first use), or **pdfLaTeX and BibTeX** with the packages loaded by the manuscript. **Poppler** (`pdftotext`) is needed for source-package verification. The builder regenerates `main.pdf` and `main.bbl`, checks for undefined references, missing citations, and overfull boxes, and removes TeX intermediates. Edit bibliography entries in `submission/refs.bib`; do not edit the PDF or generated numerical tables independently.

The build prefers Tectonic when available. Set `TECTONIC` to select its executable, or `TECTONIC_ONLY_CACHED=1` to use an already populated offline cache. The source-package check below builds and verifies a temporary archive; running `python code/scripts/package_arxiv.py` without `--check` retains a source ZIP and checksum in `submission/dist/` for manual submission.

Check the supplied folder, including source snapshots, follow-up protocol and ledgers, every saved transport coupling, bibliography, local page links, and file hashes:

```sh
python code/scripts/check_release.py
```

The script's historical filename is retained for compatibility. It checks local files and does not create a repository, tag, or release. Its [file inventory](code/results/aggregate/release-manifest.json) is a SHA-256 record of the supplied artifacts. After intentional source or documentation edits and regeneration of affected outputs, refresh it with `python code/scripts/check_release.py --write-manifest`. Hash agreement checks identity; the independent record checks assess numerical consistency. Run `python -m gcqaoa.verify` for the full canonical record reconstruction and `python code/scripts/package_arxiv.py --check` for an independently compiled source package.

Strict follow-up and transport trajectory replays require the recorded OpenBLAS backend; matching Python package versions is insufficient. On Apple silicon, create the explicit replay environment, then run the strongest check:

```sh
conda env create --prefix ./code/.replay-env --file code/environment-replay.yml
conda activate ./code/.replay-env
python -m pip install -r code/requirements-lock.txt
python -m pip install --no-deps -e code
python code/scripts/check_release.py --full --replay-solvers
```

`--full` includes the tests, canonical reconstruction, every follow-up stochastic run, and source-package compilation; `--replay-solvers` also compares all transport solver starts. These flags retain strict numerical tolerances and report backend-sensitive changes as failures. See the [backend replay notes](code/docs/PROVENANCE.md#numerical-backend-and-solver-replay).

The same regeneration workflow is available as `make -C code reproduce PYTHON=python`. Figures and numerical text follow one chain:

```text
recorded experiments → full-precision summaries → tables and figures → main.pdf
```

The generated results on `index.html` use the same records and statistical routines as the paper. To preview the page locally:

```sh
python -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/`. All project-artifact links are relative and also work when the HTML file is opened directly.

## Run new experiments

```sh
python code/scripts/run_study.py --config code/configs/paper.json
```

This writes a new execution path. Substitute its printed identifier below:

```sh
study_data="code/results/executions/EXECUTION_ID/study.json.gz"
python -m gcqaoa.verify --data "$study_data"
python code/scripts/generate_graphs.py --data "$study_data"
python code/scripts/analyze_results.py --config code/configs/paper.json --data "$study_data"
python code/scripts/make_figures.py --config code/configs/paper.json --data "$study_data"
```

With `--data`, outputs stay beside the selected execution. Omitting it selects the supplied canonical study and its manuscript outputs. The report command checks the configuration against the records; it does not run an experiment. New findings require scientific review before replacing the paper's interpretation.

Repeat the separate follow-up and transport experiments using unused output paths:

```sh
mkdir -p code/results/executions
python code/scripts/run_followup.py --output code/results/executions/followup-repeat.json.gz
python -m gcqaoa.transport_audit --output code/results/executions/transport-repeat.json.gz
```

The recorded canonical experiment took approximately 313 seconds on arm64 macOS with Python 3.13.12; the recorded smoke run took 3.3 seconds. These are measurements from that environment, not runtime guarantees. The canonical bundle is about 11 MB compressed and 64 MB as JSON, before Python-object overhead. The three supplied bundles total about 17 MB compressed. Exact simulation and MaxCut enumeration scale exponentially: the canonical graphs have at most 12 vertices and the follow-up at most 14. No GPU or quantum device is required. Numerical backends can change nearly tied donor fits and optimizer trajectories; [environment.json](code/results/aggregate/environment.json) records the supplied environment and sensitivity checks.

## Evidence and scope

| Artifact | Contents |
| --- | --- |
| [Canonical study](code/results/study.json.gz) | 4,032 runs: 24 test graphs, two depths, two shot budgets, three shot repetitions, 14 policies; a separate 16-graph donor bank. |
| [Follow-up](code/results/followup.json.gz) | Two separately seeded bank/validation/test panels; 576 runs including 448 test runs on 16 graphs at depths 2 and 3. |
| [Transport audit](code/results/transport.json.gz) | All 384 canonical target–donor pairs, with longer runs, regularization changes, multistarts, and a conditional-gradient comparison. |
| [Graph inputs](code/data/graphs.jsonl) and [splits](code/data/splits.json) | Exact edges, graph identities, seeds, and canonical bank/test membership. |
| [Reference bank](code/data/reference_bank.json) | Donor parameters and best-found objective records. |
| [Summaries](code/results/aggregate/summary.csv) and [paired comparisons](code/results/aggregate/paired_comparisons.csv) | Full-precision aggregate data behind the presentation. |

Online budgets charge initialization measurements, model fitting, rejected trials, and repeated acceptance measurements. Independent output validation is additional: 16,384 shots per canonical output and 8,192 per follow-up output. Shots, objective-estimation calls, simulation batches, and classical preprocessing are separate resources. A simulation batch is not a measured hardware job.

A signed reference gap is `objective - best_found_reference_objective`. A negative gap improves on a finite multistart reference, which is not a certified QAOA optimum. The canonical target is a gap at most 0.02; the follow-up also uses tighter targets. Exact objectives and hitting times are post-run diagnostics and do not guide online proposals or acceptance.

Graph repetitions, donor banks, and shot repetitions are different sampling units. Canonical intervals resample 24 graph means. Follow-up intervals resample graphs within each of two fixed banks; they do not estimate uncertainty across a population of banks. One follow-up test graph is isomorphic to a canonical test graph; the paper includes an exclusion sensitivity analysis. The canonical design has no separately tuned validation split.

The local graph-transfer certificate is valid under its stated hypotheses but saturates on all depth-2 target–donor pairs. Floating-point near-ties change the recorded Local TV donor on three graphs; the paper explains why that row differs from Fixed donor. Feasible transport couplings do not certify convergence or global optimality. The shapes are deterministic search geometry, not calibrated distributions over optimal parameters. The [method](code/docs/METHOD.md) and [evidence guide](code/docs/EVIDENCE.md) specify these distinctions and the limits of the acceptance guarantee.

## Folder map

```text
README.md                 Overview and runnable instructions
index.html                Local project page with generated results
code/
  src/gcqaoa/             Circuit, search, experiments, analysis, verification
  configs/                Smoke, pilot, canonical, and follow-up settings
  data/                   Graphs, reference bank, and exact source snapshots
  results/                Complete evidence bundles and aggregate outputs
  scripts/                Reproduction and manuscript-build commands
  tests/                  Independent scientific and provenance checks
  docs/                   Method, evidence, provenance, and attribution
submission/
  main.tex, main.pdf      Authoritative manuscript and matching PDF
  refs.bib, main.bbl      Editable and generated bibliography
  figures/, tables/       Generated inputs used by the manuscript
  LICENSE                 Manuscript and figure rights
```

Use `python code/scripts/export_run.py --list` to list canonical run identifiers and `--run-id ID` to export one run as configuration, metadata, events, and summary files. New executions write individual records during execution, retaining completed observations after ordinary interruptions. An interrupted execution is not a complete comparison.

## Relation to the earlier preprint

This manuscript reconstructs and replaces the earlier learned-Gaussian interpretation with graph transfer and finite-shot search. The earlier neural-network speedup, calibration, and query-efficiency claims are not reproduced. The [historical arXiv v1](https://arxiv.org/abs/2604.24803v1) does not contain these reconstructed results. The [evidence guide](code/docs/EVIDENCE.md) identifies the supporting records and their scope; this local folder does not imply that a replacement has been announced.

## Citation, dependencies, and rights

For the historical preprint:

```bibtex
@misc{huynh2026graphconditioned,
  author = {Huynh, Molena},
  title = {Query-Efficient Quantum Approximate Optimization via Graph-Conditioned Trust Regions},
  year = {2026},
  eprint = {2604.24803},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2604.24803v1}
}
```

For the reconstructed results, also identify this manuscript revision and the study-bundle SHA-256 recorded in the [figure provenance](code/results/aggregate/figure_manifest.json). The historical citation alone does not identify this evidence.

[Scientific attribution and dependency notices](code/docs/ATTRIBUTION.md) explain the implemented methods and external libraries. The [Cartan quantum synthesizer](https://github.com/kemperlab/cartan-quantum-synthesizer) inspired the compact research presentation; no code or visual assets from that project were reused.

Copyright © 2026 Molena Huynh. All rights reserved. The original code, manuscript, figures, documentation, and data compilation are publicly inspectable but are not offered under an open-source or other reuse license. See [code/LICENSE](code/LICENSE) and [submission/LICENSE](submission/LICENSE). Existing permissions for earlier separately licensed versions and external dependencies remain unaffected.
