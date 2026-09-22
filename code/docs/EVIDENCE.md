# Evidence and interpretation

The numerical claims are supported by three preserved compressed JSON bundles. Each contains its configuration or protocol, environment information and the records needed for its documented audit. The [method specification](METHOD.md) maps those records to the implementation. The [root README](../../README.md) gives commands for checking the evidence and regenerating the manuscript artifacts.

## Data inventory

| Evidence | Experimental units and scope | Contents |
| --- | --- | --- |
| [`study.json.gz`](../results/study.json.gz) | 16 bank graphs at 8/10 vertices; 24 test graphs at 10/12 vertices; depths 1/2; 14 methods; 3 shot seeds; caps 32,768/131,072; 4,032 optimization runs | Graphs, exact MaxCut values, best-found donor and target references, starts and termination states, priors and couplings, evaluations, decisions, incumbents, returned vectors, exact diagnostics and final-validation moments |
| [`followup.json.gz`](../results/followup.json.gz) | Two seeded panels, each with 8 bank, 4 validation and 8 test graphs; test sizes 12/14; depths 2/3; 7 methods; 2 shot seeds; cap 65,536; 128 tuning and 448 test runs | Full graph schedule; saved exact reference evaluations; selected settings and selection chronology; trial records, validation measurements and multiple target-gap diagnostics |
| [`transport.json.gz`](../results/transport.json.gz) | All 24-by-16 canonical target/donor pairs; 384 pairs; longer entropic solves, three regularizations, multistart and independent conditional-gradient comparisons | Couplings, scores, marginal feasibility, convergence residuals or gaps, selected starts, changed donors, explicit valid-target sets and anchor-only quality summaries |

The graph families are connected Erdos-Renyi, 3-regular, Barabasi-Albert and Watts-Strogatz instances. The canonical graph-generation configuration and the follow-up configuration are in [`configs/`](../configs). The canonical graph, split and donor exports are [`graphs.jsonl`](../data/graphs.jsonl), [`splits.json`](../data/splits.json) and [`reference_bank.json`](../data/reference_bank.json); these are views of the saved bundle rather than additional experiments.

All results use ideal statevector simulation with finite-shot multinomial observations. There are no physical-device data, device-noise experiments or large-instance scalability measurements. The reference optimum for each depth is the best point found by six BFGS starts. Its residual suboptimality is unknown.

## Metrics and accounting

| Reported quantity | Definition and interpretation |
| --- | --- |
| Objective | `f = -E[cut]/number_of_edges`; smaller is better |
| Signed reference gap | `f(returned) - f(best_found_reference)`; may be negative when the run improves on the reference |
| Deficit in percentage points | `100 * signed_reference_gap`; it is an edge-normalized expected-cut difference, not a relative percentage loss and not a certified optimality gap |
| Expected approximation ratio | `E[cut]/exact_MaxCut`; its denominator is the classical graph optimum |
| Initial quality | Exact diagnostic objective of the initial anchor or schedule; no target-graph optimization shots are needed to return that vector |
| Incremental refinement | Difference between initial and returned exact objectives, considered separately from donor quality |
| Optimization shots | Sum of the actual model, acceptance or baseline measurements recorded by `ShotOracle` |
| Calls / jobs | One sampled parameter-point estimate / one submitted simulated batch; no device latency is modeled |
| Final validation | 16,384 additional shots per canonical output; 8,192 per follow-up output; one additional call and job, with a generator separate from optimization |
| First target hit | First saved incumbent whose exact diagnostic objective is no worse than the best-found reference plus the specified gap; zero if the initial anchor qualifies |
| Capped target cost | First-hit optimization shots for successes, and the full optimization cap for failures |

First-hit cost is a retrospective exact-simulator diagnostic. It is neither an online stopping rule nor a finite-shot certificate that a hardware output meets the target. A run can already satisfy the target at its initial point yet continue spending its optimization budget. Thus zero capped target cost does not imply that the recorded optimization run used no measurements.

The canonical target gap is `0.02`; the follow-up records `0.002`, `0.005`, `0.01` and `0.02`. Final-validation observations do not replace exact diagnostics in the reported quality tables. They remain independently generated measurements of the returned output. Offline donor and target reference evaluations, validation tuning, and transport preprocessing are reported separately because exact classical objective calls have no assigned hardware-shot conversion.

## Supported findings

At depth two and the 131,072-shot cap, descriptor initialization improves mean edge-normalized expected cut by 1.05 percentage points relative to the fixed donor, with an unadjusted paired graph-bootstrap interval of 0.65 to 1.51. Relative to random initialization, the difference is 13.95 points, with interval 12.23 to 15.78. These are comparisons of the complete recorded policies; the refinement audit determines the source of their quality.

The nine undisplaced donor-based local-model policies accept **zero refinements in 2,592 runs**, while spending **196,341,504 optimization shots**. Their returned parameters equal their initial anchors. Descriptor anchor-only quality is therefore identical to descriptor local-model quality in these runs, with zero refinement calls. At the larger depth-two cap, mean signed deficits are 0.38 points for the descriptor anchor, 0.90 for the GW anchor and 0.11 for SPSA. This evidence supports transfer of useful initial parameters and identifies ineffective refinement under the recorded acceptance policy. It does not establish an efficiency advantage for the complete shaped policy.

Across all canonical local-model variants, 240 of 21,484 trial decisions are accepted and 12,763 remain unresolved. The accepted steps arise in the random-start, displaced-prior and shuffled-reference controls. There are 144 unresolved trials without any acceptance samples because the fixed-shots batch is unaffordable at the smaller cap. The current protocol's conservative acceptance behavior is a measured limitation, not evidence that useful improvements are absent.

`report.acceptance_margin_diagnostics` recomputes exact objectives at the saved proposal endpoints. Among 16,299 trials of the eight donor policies with adaptive acceptance sampling, 2,898 have a positive true decrease, using a `1e-12` positivity tolerance. Of the 16,299 trials, 15,852 have true acceptance margin `h = f(center) - f(trial) - eta * predicted <= 0.005`; this includes 2,451 of the improving proposals. For this margin range, the manuscript's finite-look proposition bounds the conditional probability of accepting any given trial by `3.4792472090386477e-6`, even if all four planned looks can be afforded. This is a conditional probability bound derived from the observation model, not an observed acceptance frequency or an estimate from independent trial records. The fixed-shots control is excluded because it uses a different confidence penalty. These additional exact evaluations are retrospective analysis and never enter optimizer decisions.

All descriptor initializations meet the canonical 0.02 gap on the 24 test graphs at each depth. Repeating each anchor for three shot seeds yields 72/72 recorded successes but still only 24 graphs. The initialization-only rows in the canonical table are derived directly from the saved anchor objectives; they are not extra independently executed experiments. They retain offline donor construction and any independent output-validation cost.

The follow-up tests deeper circuits, tighter targets, explicit initialization-only and fixed-schedule controls, and a variance-sensitive acceptance comparator. Bernstein shaping accepts one step among 32 depth-two runs and three steps among 32 depth-three runs. The corresponding isotropic counts are zero and one. At depth three, mean deficits are 0.77 points for initialization alone, 0.73 for isotropic Bernstein, 0.66 for shaped Bernstein, 0.41 for SPSA and 0.57 for COBYLA. The shaped-versus-isotropic paired changes are small: 0.03 points at depth two and 0.08 at depth three. These results demonstrate occasional refinement, without establishing broad superiority to the conventional baselines.

The follow-up tuning candidates all tie exactly on their finite-shot validation scores at both depths. Candidate order selects 256 model shots and initial radius 0.1. Tuning consumes 8,170,240 optimization shots plus 1,048,576 validation shots, and supplies no evidence that its selected candidate is better than the other grid choices. The 448 test runs consume another 18,250,752 optimization shots plus 3,670,016 final-validation shots. All follow-up reference calculations use 97,183 exact objective calls, of which 40,409 construct bank references.

The original structural solver leaves 253 of 384 pairs above its coupling-change tolerance despite satisfying marginal constraints. With longer iterations at regularization 0.05, three comparisons remain unconverged and one of 24 donors changes. Independent multistart conditional-gradient optimization changes 21 of 24 donors. At regularization 0.02, 99 selected pairwise candidates fail feasibility and only one target has a complete feasible selector. Feasibility, convergence and useful donor selection must be assessed separately. Quality means on different valid-target subsets cannot be interpreted as paired selector comparisons.

Local-neighborhood total variation saturates on 301/384 depth-one and 384/384 depth-two pairs. At depth two, the Local TV versus Fixed donor difference is caused by floating-point donor selection in three targets, as documented in [METHOD.md](METHOD.md). It provides no structural-selection evidence.

## Independence and uncertainty

The canonical experiment has one fixed bank and 24 held-out test graphs. Graphs are pairwise nonisomorphic across its bank and test splits. The follow-up has 16 test graphs across two separately seeded panels and enforces pairwise nonisomorphism among its own bank, validation and test graphs. One follow-up test graph is isomorphic to a canonical test graph, leaving 15 additional structural test instances; there is also one bank-to-bank overlap between studies. The generated manuscript reports a sensitivity analysis excluding the overlapping test graph, without refitting or rerunning methods. The follow-up was designed after inspecting the canonical experiment and is not a preregistered independent confirmation of its full procedure.

Canonical intervals first average the three shot repetitions within each graph, then use 10,000 percentile bootstrap resamples of the 24 graph means with seed 5713. Paired comparisons bootstrap graph-level differences. The full-precision [`summary.csv`](../results/aggregate/summary.csv) and [`paired_comparisons.csv`](../results/aggregate/paired_comparisons.csv) identify the experimental selection and sign convention; pairwise CSV effects are `method_a - method_b`. A positive objective difference favors method B, whereas a positive approximation-ratio difference favors method A. Manuscript expected-cut effects use the explicitly stated improvement direction.

Follow-up intervals average two shot repetitions within each graph and resample graphs separately within each of the two panels, preserving eight test graphs per panel. They are conditional on those two fixed banks. Two panels do not estimate uncertainty over a population of reference banks. Panel differences jointly vary bank graphs, validation graphs, test graphs and random streams, so they cannot be attributed to bank variability alone.

All intervals are descriptive, unadjusted 95% percentile intervals. They do not provide simultaneous coverage across the many method/depth/budget comparisons. Random-generator initial states are matched across methods and, in the canonical study, across budgets. Final-validation seeds are separate from the optimization stream but matched across methods. Method outputs and validation outcomes therefore should not be treated as independent samples when making comparisons. Repeated shot seeds and repeated uses of an unchanged anchor add no independent graphs.

## Reproduction and limits of verification

`artifacts.py` derives the graph exports and graph-level statistics. `report.py` derives the figures, manuscript numerical paragraphs and tables, and the generated result block in `index.html`. [`figure_manifest.json`](../results/aggregate/figure_manifest.json) records the selected input bundles, analysis-source hashes, graph identities and resampling settings. Edit generators when a derived artifact needs correction; do not hand-adjust a reported number to force agreement.

The scientific verifier reconstructs saved-model and acceptance calculations and recomputes exact objective diagnostics. Follow-up replay regenerates each stochastic run from its saved seed and compares deterministic observations, decisions and incumbents, excluding elapsed times. The transport verifier recomputes distortion directly from each saved coupling and checks feasibility and gap diagnostics. The tests additionally compare the circuit with independent dense operators, test analytic single-edge behavior, compare QR fitting with an SVD solution, and check atomic budgets and pooled variance.

Hashes establish input identity; tests and deterministic replay establish specified computational checks. None establishes asymptotic scaling, performance on unseen graph families, robustness to device noise, a globally optimal QAOA donor, or a global optimum of the nonconvex transport problem. Those conclusions require additional evidence.

The version-pinned pip environment validates the saved scientific records and regenerates the presentation. Strict stochastic-trajectory and transport-solver replay also require the recorded numerical backend. Apple Accelerate can change finite-precision search paths and transport feasibility relative to the recorded OpenBLAS runs. The [replay environment](../environment-replay.yml) and [backend notes](PROVENANCE.md#numerical-backend-and-solver-replay) separate this stronger check from source identity and saved-record validation.
