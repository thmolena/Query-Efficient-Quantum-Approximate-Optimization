# Source provenance

The three historical experiment bundles retain their original bytes, execution identifiers, source hashes, and recorded environment information. The refinement and mechanism experiments are separately dated bundles with embedded numerical sources. Evidence identity can be checked in a downloaded folder without `.git` and after the project is placed in a new repository.

[The snapshot manifest](../data/source_snapshots/manifest.json) binds each compressed bundle to its archived source files:

| Bundle | Recorded source scope | Verification |
| --- | --- | --- |
| `results/study.json.gz` | Five Python modules and `configs/paper.json` | `python -m gcqaoa.verify` |
| `results/followup.json.gz` | Seven Python modules, its runner, and `configs/followup.json` | `python scripts/run_followup.py --verify` |
| `results/transport.json.gz` | Four Python modules | `python -m gcqaoa.transport_audit --verify` |

Run these commands from `code/` after installing the project as described in the root README. `make check` also checks all snapshot files and their links to the unchanged bundles.

Each directory under `data/source_snapshots/` is named by the SHA-256 of its bundle's `source_sha256` mapping, serialized as sorted compact JSON. It contains exactly the source files named in that mapping, with the same paths relative to `code/`. Every file is checked against the hash already recorded in the experiment bundle; missing, extra, changed, or symbolic-link source files fail verification. The original canonical study's Git commit and dirty-state fields remain historical execution metadata. No retained commit, branch name, remote, or Git executable is required to validate the archived bytes.

The verifier first accepts an exact match to the current source files. This supports future executions made with edited code. When current bytes differ, it verifies the complete archived source set and uses the archived configuration for configuration-identity checks. Archives are read as data and are never imported or executed automatically. Keep the supplied snapshots when editing the current implementation; preserve the matching source bytes separately for any new execution that needs to survive later code changes.

A source-identity check establishes that the recorded source files remain available byte for byte. It does not establish equivalence between old and new implementations, capture dependencies outside the originally recorded source set, or replace scientific replay. The scientific verifiers separately check saved observations, decisions, accounting, graph inputs, and derived results. The follow-up command offers `--verify --full`, and the transport command offers `--verify --replay`, to rerun their recorded stochastic or solver procedures; recorded wall times are not expected to reproduce exactly. Transport summary verification requires exact graph identities, counts, and solver-status fields, while allowing at most `1e-11` percentage points of absolute rounding difference in freshly recomputed deficits (`1e-13` in the normalized objective). This handles BLAS-dependent floating-point summation without changing the saved data.

## Revised refinement execution

`results/refinement.json.gz` is a separate execution, linked by SHA-256 to the unchanged `followup.json.gz`. It embeds the precise numerical source text and hashes, protocol, frozen plan, prior anchors, graph identities, and its environment. Historical baseline rows include their source-run indices and are compared against the original bundle. New optimizer runs, model diagnostics, and endpoint replay samples carry separate identities. Proposal records are frozen and hashed before acceptance replay starts.

Run `python scripts/run_followup.py --revision --verify` to validate the new bundle. Add `--full` to regenerate every revised optimizer, affine diagnostic, and endpoint replay with the current matching numerical implementation and backend. Embedded source text preserves identity without requiring extra snapshot directories; source hashes do not themselves prove statistical validity or reproduce numerical observations. The revision records are not substitutes for the historical failures.

The current `refine` routine also exposes controlled fixed-shot allocation and pure unresolved continuation. Its default output is checked against the implementation embedded in the preserved refinement bundle, comparing every deterministic field. These optional controls do not relabel or replace the earlier execution.

## Mechanism execution and prediction freeze

`results/mechanism.json.gz` links by SHA-256 to all four earlier bundles. It embeds its numerical sources, configuration, environment, graph instances and fixed-bank priors. The planned 256 optimizer specifications hold anchor, shape, Bernstein rule, seed and budget fixed across the sampling-by-stopping factorial. Its separate model grid contains 320 prediction cells and 1,920 shot-count conditions; these counts are protocol identities, not claims of predictive accuracy.

The runner writes and hashes its plan before prediction. It then computes exact simulator inputs and Gaussian proposal forecasts, writes their hash and freeze timestamp, and only afterward draws the independent observations that test them. Fresh graphs are generated from a fixed seed and excluded by exact isomorphism checks against earlier bank, validation and test instances. Predictions and observations use distinct seed namespaces. Exact information about the fresh target is used by the offline predictor; this is neither outcome-based tuning nor an implementable measurement-only controller.

From `code/`, run:

```sh
python -m gcqaoa.mechanism --verify
python -m gcqaoa.mechanism --verify --full
```

The ordinary check verifies bundle/source identities, chronology, complete experiment grids, reconstructed gradient maps and variance propagation, controlled-run accounting, and online/extended power recomputed from preserved endpoint streams. Full replay additionally regenerates fresh graphs and priors, frozen predictions, independent model samples, schedule proposals, and all controlled optimizers. It compares deterministic fields and excludes measured elapsed times. Before full replay, every live numerical source must match its recorded hash; otherwise it raises a source-drift error. Embedded snapshots remain data and are not executed automatically. Matching package names alone does not ensure a matching numerical backend.

For a new execution, use an unused path rather than replacing the supplied evidence:

```sh
mkdir -p results/executions
python -m gcqaoa.mechanism --output results/executions/mechanism-repeat.json.gz
python -m gcqaoa.mechanism --verify --output results/executions/mechanism-repeat.json.gz
```

The runner refuses an existing output or pending-evidence path. It writes stage checkpoints and marks orderly interruptions; an interrupted bundle is not a complete result. A repeated execution has its own dates, source snapshot and measurements even when its deterministic observations reproduce the earlier run. Its output does not automatically replace manuscript inputs.

## Numerical backend and solver replay

The canonical environment record and each additional bundle identify their own numerical environment. The canonical record uses Apple Accelerate; the additional transport study was produced with Miniforge/OpenBLAS. A standard pinned pip installation can validate every saved coupling, its direct distortion, feasibility, solver status, and the resulting summaries. It need not reproduce the same nonlinear solver trajectory.

`python scripts/check_release.py` verifies source identities, follow-up/refinement/mechanism protocol records, saved transport couplings, derived artifacts, and manuscript dependencies. `--full` adds tests, canonical reconstruction, follow-up/refinement/mechanism replay, and source-package compilation. Each study's recorded source and backend requirements still apply; an environment reproducing a historical study need not reproduce a newer one. The additional `--replay-solvers` flag reruns every transport solver start and strictly compares all deterministic fields, retaining the original tolerances. Use it with the transport study's OpenBLAS environment. The ordinary record check does not claim a successful solver-trajectory replay.

For a concrete backend sensitivity witness, the entropic solver at regularization 0.02 for target `test_erdos_renyi_10_0` and donor `bank_regular_10_0` has saved distortion 0.07163402430075938. Repeating it with the same NumPy/SciPy versions under Apple Accelerate yields 0.1106302739610631. This is a materially different coupling trajectory, not floating-point summary rounding, and the strict replay correctly rejects it. The raw records are retained. Different-backend experiments should be recorded as separate runs.

A follow-up witness is validation run 107 (`panel1_validation_barabasi_albert_12`, depth 3, candidate 1, shaped Bernstein). At trial 3, the saved predicted decrease is about 0.01353947 and an Accelerate replay gives 0.01825310, after a changed interpolation point. Both runs leave the trial unresolved and return the same objective to rounding precision, but their recorded trajectories differ. The strict comparison correctly fails; an unchanged final score does not prove a reproduced trajectory.
