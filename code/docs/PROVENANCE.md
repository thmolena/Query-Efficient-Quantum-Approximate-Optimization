# Source provenance

The three supplied experiment bundles retain their original bytes, execution identifiers, source hashes, and recorded environment information. Their verification works in a downloaded folder without `.git` and after the project is placed in a new repository.

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

## Numerical backend and solver replay

The canonical environment record and each additional bundle identify their own numerical environment. The canonical record uses Apple Accelerate; the additional transport study was produced with Miniforge/OpenBLAS. A standard pinned pip installation can validate every saved coupling, its direct distortion, feasibility, solver status, and the resulting summaries. It need not reproduce the same nonlinear solver trajectory.

`python scripts/check_release.py` verifies the supplied source identities, follow-up protocol and ledgers, all saved transport couplings, derived artifacts, and manuscript dependencies. `--full` adds tests, canonical reconstruction, full follow-up trajectory replay, and compilation of the manuscript source package; full follow-up replay also requires its recorded OpenBLAS backend. The additional `--replay-solvers` flag reruns every transport solver start and strictly compares all deterministic fields, retaining the original numerical tolerances. Use it with the transport study's OpenBLAS environment. The ordinary record check does not claim a successful solver-trajectory replay.

For a concrete backend sensitivity witness, the entropic solver at regularization 0.02 for target `test_erdos_renyi_10_0` and donor `bank_regular_10_0` has saved distortion 0.07163402430075938. Repeating it with the same NumPy/SciPy versions under Apple Accelerate yields 0.1106302739610631. This is a materially different coupling trajectory, not floating-point summary rounding, and the strict replay correctly rejects it. The raw records are retained. Different-backend experiments should be recorded as separate runs.

A follow-up witness is validation run 107 (`panel1_validation_barabasi_albert_12`, depth 3, candidate 1, shaped Bernstein). At trial 3, the saved predicted decrease is about 0.01353947 and an Accelerate replay gives 0.01825310, after a changed interpolation point. Both runs leave the trial unresolved and return the same objective to rounding precision, but their recorded trajectories differ. The strict comparison correctly fails; an unchanged final score does not prove a reproduced trajectory.
