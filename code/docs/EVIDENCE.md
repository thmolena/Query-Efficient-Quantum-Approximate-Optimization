# Evidence and interpretation

The evidence consists of three preserved historical compressed JSON bundles, the refinement revision, and a separately dated mechanism experiment. Each retains its protocol, environment information and records for its documented audit. The [method specification](METHOD.md) maps those records to the implementation. The [root README](../../README.md) gives verification and regeneration commands.

## Data inventory

| Evidence | Experimental units and scope | Contents |
| --- | --- | --- |
| [`study.json.gz`](../results/study.json.gz) | 16 bank graphs at 8/10 vertices; 24 test graphs at 10/12 vertices; depths 1/2; 14 methods; 3 shot seeds; caps 32,768/131,072; 4,032 optimization runs | Graphs, exact MaxCut values, best-found donor and target references, starts and termination states, priors and couplings, evaluations, decisions, incumbents, returned vectors, exact diagnostics and final-validation moments |
| [`followup.json.gz`](../results/followup.json.gz) | Two seeded panels, each with 8 bank, 4 validation and 8 test graphs; test sizes 12/14; depths 2/3; 7 methods; 2 shot seeds; cap 65,536; 128 tuning and 448 test runs | Full graph schedule; saved exact reference evaluations; selected settings and selection chronology; trial records, validation measurements and multiple target-gap diagnostics |
| [`transport.json.gz`](../results/transport.json.gz) | All 24-by-16 canonical target/donor pairs; 384 pairs; longer entropic solves, three regularizations, multistart and independent conditional-gradient comparisons | Couplings, scores, marginal feasibility, convergence residuals or gaps, selected starts, changed donors, explicit valid-target sets and anchor-only quality summaries |
| [`refinement.json.gz`](../results/refinement.json.gz) | 256 optimizer runs on the 16 follow-up targets; depths 2/3; two shot seeds; isotropic/shaped geometry crossed with Hoeffding/Bernstein acceptance | Complete revised-policy ledgers, historical comparator links, affine diagnostics, frozen proposals and independent endpoint replay streams |
| [`mechanism.json.gz`](../results/mechanism.json.gz) | 256 controlled optimizer runs on existing targets; model predictions on eight existing and eight fresh graphs; 320 prediction cells and 1,920 shot-count cells | Sampling-by-stopping attribution, frozen exact-information predictions, 128 sampled fits per shot cell, actual-schedule proposals, and four-/five-look power re-analysis |

The graph families are connected Erdos-Renyi, 3-regular, Barabasi-Albert and Watts-Strogatz instances. The canonical graph-generation configuration and the follow-up configuration are in [`configs/`](../configs). The canonical graph, split and donor exports are [`graphs.jsonl`](../data/graphs.jsonl), [`splits.json`](../data/splits.json) and [`reference_bank.json`](../data/reference_bank.json); these are views of the saved bundle rather than additional experiments.

All results use ideal statevector simulation with finite-shot multinomial observations. There are no physical-device data, device-noise experiments or large-instance scalability measurements. The reference optimum for each depth is the best point found by six BFGS starts. Its residual suboptimality is unknown.

## Revision evidence

[`refinement.json.gz`](../results/refinement.json.gz) contains 256 fresh optimizer runs: all 16 follow-up targets at depths 2/3, two shot repetitions, and the complete isotropic/shaped by Hoeffding/Bernstein factorial. All use the saved descriptor anchor and previously validation-selected settings with a 65,536-shot cap. This is an exploratory intervention on existing targets after inspecting the earlier results, not independent confirmation on unseen targets. Each output receives 8,192 separate validation shots.

The same bundle records exact-data and sampled affine-model diagnostics, gradient errors, frozen QAOA proposals, and independent endpoint replays. Exact values are retrospective diagnostics and never enter online decisions. The protocol, source hashes and embedded source text identify this execution without modifying historical source snapshots. Graph-level intervals remain conditional on the two fixed banks. Neither deeper/somewhat larger instances nor two panels with different targets isolate bank robustness, structural-shift robustness, or hardware behavior.

All 256 revised runs terminate as `resolution_limited`: 230 reach the prescribed acceptance-look cap and 26 cannot afford the next look. A policy look cap is distinct from exhaustion of the total budget. Across 288 fitted proposals, only 31 runs fit at a changed radius (32 fits), and 28 runs use more than 256 shots per model point (28 fits). Both Hoeffding shapes retain the initial radius throughout. The Bernstein isotropic policy has 10/6 changed-radius runs at depths 2/3, of which 9/6 use more than 256 shots; the shaped policy has 10/5 changed-radius runs, of which 8/5 use more than 256. An allocation rule that is never activated cannot explain a run's observed benefit.

Restoring the original shaped-Bernstein comparator exposes a quality–cost tradeoff:

| Depth | Original mean gain (pp) | Revised mean gain (pp) | Revised minus original gain (pp) | Original mean shots | Revised mean shots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.03250787784 | 0.06517299438 | +0.03266511654 | 64,768 | 40,483.625 |
| 3 | 0.10899588805 | 0.03178089035 | -0.07721499770 | 64,512 | 38,268.6875 |

These are paired graph means computed from the preserved records, before display rounding. Both policies have the same additional 8,192-shot output evaluation. The revision spends fewer mean optimization shots at both depths, while its mean gain decreases at depth three. It is not a uniform quality improvement.

The main figures use bars and lines for initialization versus gain, actual incumbent trajectories, controlled-budget acceptance power, the radius/bias/variance mechanism, and attribution with fresh-graph prediction. Size/depth comparisons are supporting descriptive evidence. Other supporting figures retain transfer, expenditure including evaluation, conditioning, certificate saturation, and transport feasibility. Generated numerical text remains in `submission/tables/refinement.tex`; there is no separate revision-guide artifact.

## Controlled attribution and fresh-graph prediction

The mechanism experiment separates fixed versus radius-dependent model shots from stopping versus continuing after unresolved evidence. All four cells use the same descriptor anchor, shape, Bernstein acceptance, seeds, 65,536-shot cap and first-look reservation. Continuation holds the radius fixed and refits with fresh samples; it is intentionally distinct from the historical policy's continuation-and-contraction. Sixteen existing targets, depths 2/3 and two shot repetitions give 256 controlled runs. These targets remain exploratory; a complete factorial identifies the tested intervention effects, not performance beyond that sample.

All four policies return exactly the same parameters and objective in each of the 64 paired graph/depth/seed groups. Every policy has mean gain 0.03652668718 percentage points at depth two and 0.07227421989 at depth three. Their mean optimization costs differ:

| Allocation / unresolved policy | Depth-two shots | Depth-three shots |
| --- | ---: | ---: |
| Fixed / continue | 64,768 | 64,512 |
| Fixed / stop | 39,016 | 37,584 |
| Radius-dependent / continue | 64,775.25 | 63,877.125 |
| Radius-dependent / stop | 39,953.8125 | 38,069.625 |

Thus stopping avoids expenditure without changing returned quality in this controlled sample. The radius-dependent allocation produces no measured quality gain over fixed sampling here. The independent seed schedule makes these gains distinct from the preserved revision comparison above; they should not be substituted for those historical means.

The prediction grid uses the first panel's eight existing targets plus eight fresh graphs, at both depths, both shapes and five radii. The fresh graphs are nonisomorphic to all prior bank, validation and test graphs. They share the fixed first-panel donor bank. Predictions are frozen before new model samples. Six shot counts per prediction cell give 1,920 conditions, each with 128 independent fits. The actual schedule is measured at every radius, including its 64- and 16-shot operating points. The first 32 fitted models at each scheduled condition supply proposal-margin observations; fresh cells also have 256 independently generated Gaussian proposals in the frozen prediction.

The standard linear-estimator identity separates squared exact-data fitting bias from propagated shot variance. Observed RMS averages squared errors before taking the square root. Across the complete fresh-graph model grid, mean observed sampling MSE divided by mean predicted sampling MSE is 1.0161. For shaped models on the fresh graphs, Gaussian and observed positive-margin probabilities differ by a mean absolute 2.017 percentage points, with a graph-bootstrap interval of 1.450 to 2.524. This discrepancy combines approximation error and finite simulation noise.

The predictor uses exact simulator means, variances and a numerically checked gradient; the Gaussian positive-margin forecast additionally evaluates exact trial objectives. These are conditional offline predictions, not a finite-shot decision rule. The agreement supports this specified approximation within the ideal-circuit regime, without establishing an optimizer advantage, a hardware result, or a measurement-free predictor.

Four-look online power is recomputed from the already saved endpoint streams using the online confidence penalty. Five-look, higher-budget power remains a separate offline analysis. Each curve keeps the same equally weighted proposal cohort at every budget, with no selection based on replay success. Actual rule-implied expenditure includes unresolved cases at their affordable cap and resolved cases at their first decision. This analysis generates no new shots.

Among the 58 fixed positive-margin proposals, mean Bernstein acceptance frequency is 2.33% at the online 32,768-shot endpoint-pair cap and 28.77% at the extended 131,072-shot cap. Mean unresolved fractions are 97.67% and 71.23%, respectively. Hoeffding records zero acceptances at both caps, which does not establish zero acceptance probability. The 102 nonpositive-margin proposals form a separate fixed cohort in the power figure.

The controlled optimizers consume 13,201,786 optimization shots plus 2,097,152 independent validation shots. The separate model experiment generates 5,367,398,400 simulated observations. Prediction construction uses 43,872 exact objective calls; evaluating the 10,240 sampled scheduled proposals adds 10,240 diagnostic exact calls. These offline costs are distinct from optimizer expenditure and have no assigned hardware-shot equivalent.

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
| Final validation | 16,384 additional shots per canonical output; 8,192 per follow-up, refinement or attribution output; one additional call and job, with a generator separate from optimization |
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

Gain and model-error graph intervals are descriptive, unadjusted 95% percentile intervals. Trajectory contrasts resample whole paired graph trajectories; their bands are pointwise, not simultaneous. Mechanism RMS intervals apply the square root after each resampled mean-square calculation. Fixed-cohort acceptance-power curves instead use pointwise Hoeffding bounds over independent proposal/replay outcomes, conditional on the selected proposals; they do not estimate graph-population uncertainty. Zero acceptances among 128 repeats of one proposal does not establish zero probability: its one-sided 95% binomial upper limit is about 2.31%.

Random-generator initial states are matched across methods and, in the canonical study, across budgets. Final-validation seeds are separate from optimization but matched across methods. Shared endpoint streams across power rules and budgets likewise make their outcomes dependent. Method outputs and validation outcomes should therefore not be treated as independent samples when making comparisons. Repeated shot seeds, radii, fitted models and repeated uses of an unchanged anchor add no independent graphs.

## Reproduction and limits of verification

`artifacts.py` derives the graph exports and graph-level statistics. `report.py` derives the historical and revised figures, manuscript numerical paragraphs and tables, and the generated result block in `index.html`. [`figure_manifest.json`](../results/aggregate/figure_manifest.json) records the selected input bundles, analysis-source hashes, graph identities and resampling settings. Edit generators when a derived artifact needs correction; do not hand-adjust a reported number to force agreement.

The canonical scientific verifier reconstructs saved-model and acceptance calculations and recomputes exact objective diagnostics. Follow-up and refinement full replay regenerate stochastic runs from saved seeds and compare deterministic observations, decisions and incumbents, excluding elapsed times. The transport verifier recomputes distortion from saved couplings and checks feasibility and gap diagnostics. Mechanism verification checks identities, prediction chronology, grid completeness, fitting maps, variance propagation, accounting and re-analysis of preserved power streams; full replay also repeats graph generation, predictions, sampled fits and controlled optimizers, requiring the recorded source bytes. The tests compare the circuit with independent dense operators, test analytic single-edge behavior, compare QR with SVD, exhaustively enumerate a small binomial model's MSE, and check shared first-trial streams and unchanged historical defaults.

Hashes establish input identity; tests and deterministic replay establish specified computational checks. None establishes asymptotic scaling, performance on unseen graph families, robustness to device noise, a globally optimal QAOA donor, or a global optimum of the nonconvex transport problem. Those conclusions require additional evidence.

The version-pinned pip environment validates the saved scientific records and regenerates the presentation. Strict stochastic-trajectory and transport-solver replay also require the recorded numerical backend. Apple Accelerate can change finite-precision search paths and transport feasibility relative to the recorded OpenBLAS runs. The [replay environment](../environment-replay.yml) and [backend notes](PROVENANCE.md#numerical-backend-and-solver-replay) separate this stronger check from source identity and saved-record validation.
