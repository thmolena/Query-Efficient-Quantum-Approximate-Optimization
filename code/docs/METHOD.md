# Method and implementation

This document specifies the algorithms represented by the saved experiments. The implementation uses a small set of working modules; the links below identify their actual responsibilities. [EVIDENCE.md](EVIDENCE.md) describes the experimental units and the limits of the conclusions. Build and execution commands are in the [root README](../../README.md).

| Component | Implementation | Independent checks or saved evidence |
| --- | --- | --- |
| Circuit, cuts, shot distribution | [`qaoa.py`](../src/gcqaoa/qaoa.py): `MaxCutQAOA`, `exact_maxcut` | Dense-operator and single-edge checks in [`test_qaoa.py`](../tests/test_qaoa.py) |
| Graph generation, references, experiment schedule | [`experiment.py`](../src/gcqaoa/experiment.py): `instances`, `reference`, `_run` | Graphs, split assignments, reference starts and complete run records in `study.json.gz` |
| Donor scores, symmetry alignment, region shape | [`search.py`](../src/gcqaoa/search.py): `descriptor`, `graph_prior`, `align` | Saved priors; geometry checks in [`test_search.py`](../tests/test_search.py) |
| Local graph comparison | [`qaoa.py`](../src/gcqaoa/qaoa.py): `local_signature`, `local_tv` | Exact rooted-isomorphism and objective-bound checks in `test_qaoa.py` |
| Structural transport comparison | [`search.py`](../src/gcqaoa/search.py): `structural_score`; [`transport_audit.py`](../src/gcqaoa/transport_audit.py) | Saved couplings, marginal errors, solver residuals and direct four-index distortion checks |
| Local interpolation, finite-shot search and baselines | [`search.py`](../src/gcqaoa/search.py): `design_points`, `fit_model`, `ShotOracle`, `optimize` | Independent least-squares comparison, atomic budgets, reconstructed model and decision checks |
| Variance-sensitive follow-up | [`followup.py`](../src/gcqaoa/followup.py): `pool_moments`, `bernstein_optimize`, `select_settings` | Saved validation schedule and settings; replay and pooled-variance checks in [`test_followup.py`](../tests/test_followup.py) |
| Statistical summaries and manuscript artifacts | [`artifacts.py`](../src/gcqaoa/artifacts.py), [`report.py`](../src/gcqaoa/report.py) | Graph-level CSVs, figure input manifest and generated `submission/tables/results.tex` |
| Scientific audit | [`verify.py`](../src/gcqaoa/verify.py) | Reconstructs graph splits, objectives, shot ledgers, fitted models and acceptance decisions |

## Circuit and observation conventions

For a simple, undirected, unweighted graph with `m > 0` edges, `C` is the diagonal cut-count operator and `B = sum_j X_j`. The initial state is the uniform superposition. Each layer applies `exp(-i gamma_l C)` followed by `exp(-i beta_l B)`. Parameters are ordered `(gamma_1, ..., gamma_p, beta_1, ..., beta_p)`. The minimized objective is `f(theta) = -E[C]/m`, so a lower value is better and every shot outcome lies in `[-1, 0]`. Graph node insertion order determines computational-basis bit order.

`sample` draws multinomial counts from the exact ideal-circuit distribution aggregated by cut value. It returns a sample mean and `sample_variance / shots`, using the unbiased sample-variance denominator `shots - 1`. This is a variance estimate for the mean, not the variance of individual shots. There is no device noise model. The optimizer receives these sampled quantities; exact expectations are calculated separately for offline references and retrospective diagnostics.

The objective is invariant under an individual `2*pi` period in each gamma, an individual `pi/2` period in each beta, and simultaneous reversal of all angles. `wrap` uses the centered period cell. `align` chooses the closest representative to an anchor among the two global signs after componentwise period shifts; an exact distance tie keeps the positive-sign representative.

## Graphs, references and donor selection

The canonical configuration is [`paper.json`](../configs/paper.json). `instances` iterates split, configured family order, configured size order and replicate. The family order is Erdos-Renyi, 3-regular, Barabasi-Albert and Watts-Strogatz. Connected graphs are generated using the recorded seeds; each proposed graph is rejected if it is isomorphic to an earlier graph in either split. The preserved instance list fixes bank order. The follow-up repeats this construction within its two separately seeded panels, with a validation split and a global nonisomorphism check across those panels.

Each depth-specific donor vector is the best objective among six exact-objective BFGS starts: one declared schedule and five random starts. The canonical cap is 180 BFGS iterations per start; the follow-up cap is 120. All starts retain the returned point, objective, function-call count and termination status. These best-found QAOA references have no certified global-optimality gap. Exact MaxCut enumeration supplies a separate classical denominator for approximation ratios; it does not certify the variational optimum.

The graph descriptor contains mean degree, degree standard deviation, the 25th/50th/75th degree quantiles and mean clustering coefficient. Descriptor distances use componentwise bank standard deviations, floored at `0.1`. Alternative distances are all zeros for the fixed-donor control, local edge-neighborhood total variation, or the returned structural transport distortions. Shuffling permutes the transport score assignments while preserving donor order and parameters.

For distances `d_i`, the implemented weight is proportional to `exp(-(d_i-min(d))/tau)`, with `tau = max(std(d), 0.001)`. The anchor is the first maximum of these computed floating-point weights, before the neighborhood filter described below. There is no numerical near-tie tolerance. The fixed-donor control therefore always selects bank entry zero. Mathematically equal but numerically unequal scores can select a different entry.

**Recorded depth-two tie artifact.** All 384 canonical depth-two local-TV comparisons saturate at one. Their floating-point sums have a maximum within-target spread of `4.440892098500626e-16`. The softmax retains enough of this variation that three targets choose a different anchor from the fixed-donor control:

| Target | Fixed donor index | Recorded local-TV donor index |
| --- | ---: | ---: |
| `test_regular_10_2` | 0 | 1 |
| `test_barabasi_albert_10_0` | 0 | 1 |
| `test_watts_strogatz_10_2` | 0 | 3 |

These are zero-based bank indices. The Local TV and Fixed donor table rows consequently differ for nine of 72 runs per budget. That difference is a rounding-induced donor selection effect and supplies no evidence that the saturated local comparison distinguishes graph structure. The records and tables preserve the policy actually executed.

## Local graph bound and structural scores

`local_signature` groups the induced radius-`p` neighborhoods of every edge by exact graph isomorphism preserving an unordered pair of marked root endpoints. The type distribution weights each edge equally. `local_tv` aligns the rooted types across graphs and computes one half of the sum of absolute frequency differences. Each edge's expected normalized cut contribution depends only on this neighborhood and lies in `[0,1]`; consequently the discrepancy between graph objectives at any common parameter vector is at most this total variation. A donor-to-target optimization bound additionally requires a bound on donor suboptimality. A best-found donor alone does not supply that assumption. Saturation at one gives no improvement over the objective's full range.

The transport score compares all-pairs shortest-path distances divided by each graph's diameter, with uniform node masses. `structural_score` initializes the coupling to the product of these masses. It performs damped fixed-point Sinkhorn updates with regularization `0.05`, at most 100 outer iterations and 500 inner iterations, damping `0.5`, outer coupling-change tolerance `1e-9` and inner row-marginal tolerance `1e-12`. The tensor entering the kernel is rounded to 12 decimals to suppress roundoff-driven symmetry breaking at the uniform start. A marginal error above `1e-8` raises an error. The returned score is the unregularized squared distortion at that coupling. Neither feasibility nor the stopping residual proves a globally optimal graph distance.

The separate transport audit increases the outer cap to 1,000, evaluates regularizations `0.02`, `0.05`, `0.1`, and compares three starts at `0.05` and three starts for an independent conditional-gradient implementation. The latter uses transportation linear programs and records a first-order gap. An infeasible comparison invalidates the complete donor selector for that target; quality is reported only on explicitly identified valid targets.

## Region shape and local interpolation

After choosing the anchor, donor angles are aligned using the exact objective symmetries. Only aligned donors within Euclidean distance one of the anchor retain weight, and their weights are renormalized. With displacements `v_i`, the regularized second moment is `C = 0.02 I + sum_i w_i v_i v_i^T`. Its log eigenvalues are centered, clipped to `[-log(2), log(2)]` and centered again. Halving these logs and exponentiating gives the eigenvalues of a symmetric positive-definite shape `S`. Thus `det(S) = 1`, `cond(S) <= 2` and `cond(S S^T) <= 4`. The physical region is `theta + radius * S u` for `||u|| <= 1`, with angles wrapped for evaluation. The matrix variable is named `B` in `search.py`; it is distinct from the circuit mixer operator.

For dimension `d = 2p`, the candidate set contains the origin, positive and negative coordinate axes, and `d+1` random unit directions. Column-pivoted QR of the transposed augmented design selects `d+1` points. The selected affine design must have full rank. The `no_repair` control uses only the random directions. Each selected point receives the configured model-shot count.

`fit_model` solves weighted least squares by QR with square-root shot-count weights and triangular solution. In these experiments there are exactly `d+1` observations for `d+1` coefficients. This is square interpolation: positive weights do not change the exact interpolant, although shot noise and the design condition number affect coefficient accuracy. The reported condition number belongs to the weighted triangular factor. It is a diagnostic, not a certificate of saved measurements.

Writing the model in local coordinates as `c + g^T u`, the trial uses `u = -g/||g||` and predicted decrease `q = ||g||`. A model with `q < 1e-12` stops without an acceptance test.

## Acceptance, budgets and stopping

The canonical adaptive policy allows at most `T = 64` trials and cumulative per-endpoint looks `s = 512, 2048, 8192, 32768`. At each trial, fresh observations of the current center and proposed point are collected in separate endpoint streams. Model observations are not reused. For `L` prescribed looks and `alpha = 0.05`, the endpoint Hoeffding radius is

```text
b_s = sqrt(log(4*T*L/alpha)/(2*s)).
```

The decrease estimate is the center mean minus the trial mean. Its interval has half-width `2*b_s`. The proposal is accepted when its lower endpoint is at least `eta*q`, with `eta = 0.1`, and rejected when its upper endpoint is strictly below `eta*q`. Otherwise the next affordable prescribed look is attempted. A decision still undecided at its sampling cap or available budget is recorded as unresolved and keeps the center. A union bound over both endpoints and all prescribed trials/looks supplies run-wise coverage at least `1-alpha`, conditional on the past before choosing each trial. This controls incorrect acceptance decisions; it does not guarantee that beneficial moves will be accepted, that the optimizer converges, or that a target will be attained.

Accepted trials multiply the radius by `1.5` up to its cap. Rejected and unresolved trials halve it down to its floor. The fixed-radius control omits this update. The fixed-shots control has only the final prescribed look, so its confidence penalty uses `L = 1`. Its two-endpoint acceptance batch costs 65,536 shots: it is unaffordable under the smaller 32,768-shot canonical cap even before model costs. The saved experiment therefore contains 144 zero-acceptance-sample unresolved decisions. This control is an intentionally disclosed complete-policy comparison at the recorded settings, not a matched-confidence test of sample allocation alone.

`ShotOracle.batch` checks the entire proposed batch before sampling. A batch that exceeds the remaining budget consumes no shots, calls or jobs. A call means one parameter-point estimate; a job means one requested batch of such estimates. Model construction is attempted before its acceptance batch, so model shots can be spent on a trial for which no acceptance look is affordable. After recording a nonflat decision and updating its radius, the optimizer stops if the remaining budget cannot pay for a new model and the first prescribed acceptance look. Otherwise it may attempt another trial even when a later look of the preceding trial was unaffordable. Budget caps are upper bounds rather than expenditure targets. The `stop` field, decision records and ledger should be interpreted together; a model-batch failure in the canonical function does not explicitly replace its initial `horizon` label.

The follow-up changes the horizon to 16 and the final look to 16,384 per endpoint. Its empirical-Bernstein comparator pools all independent endpoint batches, including the between-batch contribution to sample variance. Its endpoint radius is

```text
sqrt(2 * sample_variance * log(8*T*L/alpha) / s)
    + 7 * log(8*T*L/alpha) / (3*(s-1)).
```

This is the two-tail, two-endpoint finite-horizon use of the empirical-Bernstein bound cited in the manuscript. Acceptance and radius updates otherwise use the same comparisons. This comparator is an independent implementation of those ingredients.

## Comparators, returned points and diagnostic separation

| Policy | Initialization and shape | Returned point |
| --- | --- | --- |
| Fixed donor, descriptor, GW, Local TV | Respective anchor; isotropic or shaped region as named | Last accepted center |
| Shuffled GW | Anchor and shape from permuted score assignments | Last accepted center |
| Displaced prior | Descriptor anchor plus `1.2` in every gamma and `0.35` in every beta, then wrapped; descriptor shape | Last accepted center |
| Random start | Independently drawn centered-cell angles; identity shape | Last accepted center |
| No repair, fixed radius, fixed shots | Descriptor anchor and descriptor shape | Last accepted center |
| COBYLA | Descriptor anchor; configured starting radius; `tol = 0.002`; 1,024 shots per objective call | Evaluated point with the smallest observed sample mean, with first-occurrence tie handling |
| SPSA | Descriptor anchor; two perturbations and one post-update center per iteration, each 1,024 shots | Post-update center with the smallest observed sample mean, with first-occurrence tie handling |
| Follow-up initialization only | Descriptor anchor | Initial anchor, without optimization calls |
| Follow-up fixed schedule | Declared graph-independent schedule | Schedule, without optimization calls |

SPSA uses independent Rademacher directions, `a_k = 0.15/(k+10)^0.602`, `c_k = 0.12/k^0.101`, and wraps the update. Perturbation points are not candidates for its returned incumbent. Neither baseline uses exact objectives to choose an output. SPSA's initial point is recorded for retrospective target attainment, but it is not measured as a candidate before the first update; the baseline can therefore return a worse true point than its initial anchor. The COBYLA output is the best sampled point, which can differ from SciPy's terminal iterate.

The canonical seed is shared across methods and budgets for each graph/depth/repetition. The follow-up similarly matches initial random-generator states across methods and tuning candidates. These are paired computational comparisons, not independent streams between methods. Fresh endpoint observations within an individual acceptance trial have the independence required by its bound. Final validation uses a separate generator seed and does not choose the reported output. Across methods, final-validation seeds are again matched.

Exact reference values, exact final objective values, approximation ratios and exact incumbent target-hit diagnostics are computed outside the optimizer. Target attainment never triggers its stopping rule. The follow-up selects the common depth-specific settings by mean independent finite-shot validation objective over both validation panels and both Bernstein variants; it never uses test outcomes in that selection. All four recorded candidates tie at each depth, and the first candidate wins.
