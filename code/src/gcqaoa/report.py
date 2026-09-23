"""Regenerate all manuscript statistics and vector figures from the saved study."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

from .provenance import CANONICAL_DATA, CODE_ROOT, REPO_ROOT, digest, load_study
from .artifacts import analyze, ci, lineage, write_json
WEB_BEGIN = '<!-- BEGIN GENERATED CANONICAL RESULTS -->'
WEB_END = '<!-- END GENERATED CANONICAL RESULTS -->'
EXPERIMENTS_BEGIN = '<!-- BEGIN GENERATED ADDITIONAL RESULTS -->'
EXPERIMENTS_END = '<!-- END GENERATED ADDITIONAL RESULTS -->'
FOLLOWUP_LABELS={'initialization_only':'Descriptor anchor', 'fixed_schedule':'Fixed schedule',
                 'hoeffding_iso':'Hoeffding, isotropic', 'bernstein_iso':'Bernstein, isotropic',
                 'bernstein_shape':'Bernstein, shaped', 'spsa':'SPSA', 'cobyla':'COBYLA'}
TRANSPORT_LABELS={'entropic_0.02':r'Entropic $\epsilon=0.02$',
                  'entropic_0.05':r'Entropic $\epsilon=0.05$',
                  'entropic_0.1':r'Entropic $\epsilon=0.1$',
                  'entropic_multistart':'Entropic, three starts',
                  'conditional_gradient':'Conditional gradient, three starts'}
LABELS={
 'uniform_iso':'Fixed donor', 'random_iso':'Random start',
 'descriptor_iso':'Descriptor, isotropic', 'descriptor_shape':'Descriptor, shaped',
 'gw_iso':'GW, isotropic', 'gw_shape':'GW, shaped', 'local_iso':'Local TV',
 'shuffled_gw':'Shuffled GW', 'bad_prior':'Displaced prior',
 'no_repair':'No point repair', 'fixed_radius':'Fixed radius',
 'fixed_shots':'Fixed shots', 'cobyla':'COBYLA', 'spsa':'SPSA'}
COLORS=['#526777','#999999','#0072B2','#009E73','#E69F00','#D55E00',
        '#CC79A7','#6B5CA5','#222222']


def selected(data,method,p,budget):
    return [r for r in data['runs'] if r['method']==method and r['depth']==p and r['budget']==budget]


def graph_means(rows,field):
    return {g:float(np.mean([r[field] for r in rows if r['graph']==g])) for g in sorted({r['graph'] for r in rows})}


def paired(data,a,b,p,budget,field='objective',scale=100):
    x,y=graph_means(selected(data,a,p,budget),field),graph_means(selected(data,b,p,budget),field)
    return ci([scale*(x[g]-y[g]) for g in x])


def fmt_ci(triple):
    mean,lo,hi=triple
    return f'{mean:.2f} (95\\% interval {lo:.2f} to {hi:.2f})'


def acceptance_margin_diagnostics(data, threshold=0.005):
    """Recompute proposal margins offline; never supplies optimizer inputs.

    The true margin is f(center)-f(trial)-eta*predicted, evaluated at the
    recorded endpoints. Fixed-shot trials are counted separately because
    they use a different number of prescribed looks in the confidence bound.
    Counts are repeated trial records, not independent graph observations.
    """
    from .experiment import graph_from
    from .qaoa import MaxCutQAOA
    methods={'uniform_iso','descriptor_iso','descriptor_shape','gw_iso','gw_shape',
             'local_iso','no_repair','fixed_radius','fixed_shots'}
    settings=data['config']['optimizer']
    selected_runs=[row for row in data['runs'] if row['method'] in methods]
    graph_ids={row['graph'] for row in selected_runs}
    circuits={row['id']:MaxCutQAOA(graph_from(row)) for row in data['instances']
              if row['id'] in graph_ids}
    cache={}
    def objective(graph,theta):
        key=(graph,tuple(theta))
        if key not in cache:
            cache[key]=circuits[graph].objective(theta)
        return cache[key]
    def empty():
        return {'trials':0,'positive_true_decrease':0,'positive_margin':0,
                'margin_at_most_threshold':0,'positive_decrease_below_threshold':0}
    counts={name:empty() for name in ('all_donor','adaptive_donor','fixed_shots')}
    for row in selected_runs:
        group='fixed_shots' if row['method']=='fixed_shots' else 'adaptive_donor'
        for decision in row['decisions']:
            decrease=objective(row['graph'],decision['center'])-objective(row['graph'],decision['trial'])
            margin=decrease-settings['eta']*decision['predicted']
            for name in ('all_donor',group):
                count=counts[name]
                count['trials']+=1
                count['positive_true_decrease']+=int(decrease>1e-12)
                count['positive_margin']+=int(margin>1e-12)
                count['margin_at_most_threshold']+=int(margin<=threshold)
                count['positive_decrease_below_threshold']+=int(decrease>1e-12 and margin<=threshold)
    looks=np.asarray(settings['acceptance_looks'],dtype=float)
    radii=np.sqrt(np.log(4*settings['max_trials']*len(looks)/settings['alpha'])/(2*looks))
    upper=float(min(1.,np.exp(-looks*np.maximum(2*radii-threshold,0.)**2).sum()))
    return {'threshold':float(threshold),'positivity_tolerance':1e-12,
            'counts':counts,'exact_endpoint_evaluations':len(cache),
            'adaptive_per_trial_acceptance_bound':upper}


def website_results(data):
    """Accessible inline figure and table, using the manuscript's graph bootstrap."""
    budget=max(data['config']['budgets'])
    anchor=selected(data,'descriptor_iso',2,budget)
    initial={g:float(np.mean([r['initial_objective']-r['reference_objective']
                             for r in anchor if r['graph']==g]))
             for g in sorted({r['graph'] for r in anchor})}
    rows=[('Descriptor anchor only',ci([100*v for v in initial.values()]),0.0)]
    for method in ['descriptor_shape','gw_shape','cobyla','spsa']:
        group=selected(data,method,2,budget)
        interval=ci([100*v for v in graph_means(group,'signed_reference_gap').values()])
        rows.append((LABELS[method],interval,float(np.mean([r['shots'] for r in group]))))
    n=len(initial)
    left,right=235,680
    lower=min(0.0,min(interval[1] for _,interval,_ in rows))
    upper=max(interval[2] for _,interval,_ in rows)*1.12
    x=lambda value:left+(value-lower)/(upper-lower)*(right-left)
    lines=[WEB_BEGIN, '<figure class="result-figure">',
           '<svg viewBox="0 0 720 285" role="img" aria-labelledby="results-title results-description">',
           '<title id="results-title">Initialization and refinement at depth two</title>',
           f'<desc id="results-description">Mean signed reference deficit with 95 percent graph-bootstrap intervals over {n} graphs. Smaller is better. Exact numerical values and online shot costs follow in the table.</desc>',
           '<g fill="none" stroke="#d7ddd4" stroke-width="1">']
    # Ticks must sit at round values, not at linspace endpoints: labelling a gridline
    # drawn at 1.4868 as "1.5" makes values read off the axis disagree with the table.
    step=next(s for s in (0.05,0.1,0.2,0.25,0.5,1.0,2.0,5.0) if (upper-lower)/s<=4)
    ticks=np.arange(np.ceil(lower/step)*step,upper+1e-9,step)
    for tick in ticks:
        lines.append(f'<path d="M{x(tick):.2f} 24 V225"/>')
    lines+=['</g>', '<g font-family="system-ui,sans-serif" font-size="14" fill="#22322a">']
    for index,(label,(mean,lo,hi),_) in enumerate(rows):
        y=44+index*40
        color='#0072b2' if index<2 else '#b25123' if index==2 else '#3b6550'
        lines+=[f'<text x="8" y="{y+5}">{label}</text>',
                f'<path d="M{x(lo):.2f} {y} H{x(hi):.2f} M{x(lo):.2f} {y-5} V{y+5} M{x(hi):.2f} {y-5} V{y+5}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<circle cx="{x(mean):.2f}" cy="{y}" r="4.5" fill="{color}"/>']
    for tick in ticks:
        lines.append(f'<text x="{x(tick):.2f}" y="246" text-anchor="middle">{tick:.1f}</text>')
    lines+=['<text x="458" y="274" text-anchor="middle">Signed reference deficit (percentage points)</text>',
            '</g></svg>',
            f'<figcaption>Depth p = 2; {budget:,}-shot online cap. Points show means and bars show 95% percentile bootstrap intervals (10,000 resamples; seed 5713; n = {n} graphs). Three shot repetitions are averaged within each graph. Intervals are descriptive and unadjusted.</figcaption>',
            '</figure>', '<div class="table-scroll"><table>',
            '<caption>Canonical results and measured online costs. The descriptor anchor is evaluated before refinement.</caption>',
            '<thead><tr><th scope="col">Policy</th><th scope="col">Deficit, 95% CI (pp)</th><th scope="col">Mean online shots</th></tr></thead><tbody>']
    for label,(mean,lo,hi),shots in rows:
        lines.append(f'<tr><th scope="row">{label}</th><td>{mean:.2f} [{lo:.2f}, {hi:.2f}]</td><td>{shots:,.0f}</td></tr>')
    lines+=['</tbody></table></div>',
            '<p class="small">Deficit is the signed edge-normalized objective gap to a best-found depth-matched reference, not a certified optimum. Each optimized output also receives 16,384 independent validation shots. Anchor-only rows require no online search shots; offline donor construction and any validation cost remain separate.</p>',
            WEB_END]
    return '\n'.join(lines)


def update_website(data):
    path=REPO_ROOT/'index.html'
    content=path.read_text()
    if content.count(WEB_BEGIN)!=1 or content.count(WEB_END)!=1:
        raise ValueError('Website must contain exactly one canonical-results block')
    prefix,remainder=content.split(WEB_BEGIN,1)
    _,suffix=remainder.split(WEB_END,1)
    content=prefix+website_results(data)+suffix
    if all((CODE_ROOT/'results'/name).is_file() for name in ('followup.json.gz','transport.json.gz')):
        from .followup import load_followup
        followup=load_followup(verify=False)
        with gzip.open(CODE_ROOT/'results'/'transport.json.gz','rt') as stream:
            transport=json.load(stream)
        if content.count(EXPERIMENTS_BEGIN)!=1 or content.count(EXPERIMENTS_END)!=1:
            raise ValueError('Website must contain exactly one additional-results block')
        prefix,remainder=content.split(EXPERIMENTS_BEGIN,1)
        _,suffix=remainder.split(EXPERIMENTS_END,1)
        content=prefix+website_additional(followup,transport,data)+suffix
    path.write_text(content)


def followup_group(data, method, depth):
    return [r for r in data['runs'] if r['phase']=='test' and r['method']==method and r['depth']==depth]


def bank_conditional_ci(values, panels):
    """Percentile graph bootstrap stratified within the two fixed reference banks."""
    rng=np.random.default_rng(5713)
    total=np.zeros(10000)
    for panel in sorted(set(panels.values())):
        sample=np.array([values[g] for g in sorted(values) if panels[g]==panel],dtype=float)
        total+=sample[rng.integers(0,len(sample),(10000,len(sample)))].sum(1)
    return float(np.mean(list(values.values()))),*map(float,np.quantile(total/len(values),[.025,.975]))


def followup_ci(group, field='signed_reference_gap', scale=100):
    values={g:scale*v for g,v in graph_means(group,field).items()}
    return bank_conditional_ci(values,{r['graph']:r['panel'] for r in group})


def followup_overlap(data, canonical, split='test'):
    import networkx as nx
    from .experiment import graph_from
    original=[graph_from(r) for r in canonical['instances'] if r['split']==split]
    return sorted(r['id'] for r in data['instances'] if r['split']==split and
                  any(r['n']==len(graph) and nx.is_isomorphic(graph_from(r),graph) for graph in original))


def website_additional(data, transport, canonical):
    tests=[r for r in data['runs'] if r['phase']=='test']
    n=len({r['graph'] for r in tests})
    overlap=followup_overlap(data,canonical)
    bank_overlap=followup_overlap(data,canonical,'bank')
    lines=[EXPERIMENTS_BEGIN,
           f'<p>Two separately seeded panels add {len(tests)} held-out runs on {n} test graphs at depths 2 and 3, with a separate validation split. The bank uses 8/10-vertex graphs, validation uses 12, and tests use 12/14. Validation selects a common model-shot count and initial radius for the isotropic and shaped variance-sensitive rules before testing.</p>',
           f'<p>{len(overlap)} follow-up test graph is isomorphic to a canonical test graph; {n-len(overlap)} are structurally distinct from the canonical test set. There is also {len(bank_overlap)} bank-to-bank overlap between studies. All follow-up bank, validation, and test graphs are mutually nonisomorphic within that study. The manuscript also reports sensitivity after excluding the test overlap.</p>',
           '<div class="table-scroll"><table>',
           '<caption>Follow-up depth 3: mean signed deficit and 95% graph-bootstrap interval over 16 graph means (two shot repetitions each). The online cap is 65,536 shots; add 8,192 independent validation shots per output.</caption>',
           '<thead><tr><th scope="col">Policy</th><th scope="col">Deficit, 95% CI (pp)</th><th scope="col">Mean online shots</th></tr></thead><tbody>']
    for method in data['config']['methods']:
        group=followup_group(data,method,3)
        mean,lo,hi=followup_ci(group)
        shots=np.mean([r['shots'] for r in group])
        lines.append(f'<tr><th scope="row">{FOLLOWUP_LABELS[method]}</th><td>{mean:.2f} [{lo:.2f}, {hi:.2f}]</td><td>{shots:,.0f}</td></tr>')
    lines+=['</tbody></table></div>',
            f'<p class="small">Intervals use 10,000 graph resamples within each bank and seed 5713, conditional on the two fixed reference banks; they are descriptive and unadjusted. Two banks do not estimate uncertainty over a population of reference banks. Tuning consumes {data["tuning"]["optimization_shots"]:,} optimization shots plus {data["tuning"]["validation_shots"]:,} validation shots. Exact simulation, best-found reference construction, and solver preprocessing are separate classical costs.</p>']
    if all(len(set(scores))==1 for scores in data['tuning']['scores'].values()):
        lines.append('<p>All four tuning candidates tie on validation scores at each depth; the declared candidate order selects 256 model shots and radius 0.1. This tie provides no evidence of tuning superiority. The variance-sensitive rules refine a few held-out outputs, while SPSA and COBYLA achieve lower mean deficits.</p>')
    row=next(r for r in transport['summary'] if r['solver']=='entropic_0.05' and r['depth']==2)
    fw=next(r for r in transport['summary'] if r['solver']=='conditional_gradient' and r['depth']==2)
    lines.append(f'<p>The solver audit checks all 384 canonical graph pairs. Longer entropic runs at regularization 0.05 change {row["changed_donors"]} of {row["valid_targets"]} evaluable target donors; {row["unconverged_pairs"]} pairwise candidates remain unconverged and {row["infeasible_pairs"]} fail feasibility. An independent multistart conditional-gradient solver changes {fw["changed_donors"]} of {fw["valid_targets"]} evaluable donors. Feasibility or a small first-order gap does not certify a globally optimal graph comparison.</p>')
    lines.append(EXPERIMENTS_END)
    return '\n'.join(lines)


def followup_results(data, canonical):
    """Summarize held-out follow-up runs; validation costs stay separate."""
    tests=[r for r in data['instances'] if r['split']=='test']
    test_runs=[r for r in data['runs'] if r['phase']=='test']
    validation_runs=[r for r in data['runs'] if r['phase']=='validation']
    overlap=followup_overlap(data,canonical)
    bank_overlap=followup_overlap(data,canonical,'bank')
    lines=[r'\paragraph{Separately seeded panels test tighter targets and deeper circuits.}',
           f'The separate follow-up contains {len(test_runs)} test runs on {len(tests)} graphs from two separately seeded bank--validation--test panels, at depths 2 and 3. '
           f'Its {len(validation_runs)} tuning runs consume {data["tuning"]["optimization_shots"]:,} optimization shots and {data["tuning"]["validation_shots"]:,} independent validation shots. '
           'The selected model-shot count and initial radius are frozen before any test run and shared by the isotropic and shaped variance-sensitive policies at each depth. '
           'All methods use a 65,536-shot online cap; each output receives 8,192 additional validation shots. '
           'Table~\\ref{tab:followup} keeps initialization-only outcomes and actual online costs visible.']
    lines.append(f'Exact isomorphism checks identify {len(overlap)} follow-up test graph also present in the canonical test set, leaving {len(tests)-len(overlap)} structurally distinct test graphs. '
                 f'There is also {len(bank_overlap)} bank-to-bank overlap across the two studies. '
                 'The follow-up bank, validation and test splits remain pairwise nonisomorphic within the new study. ')
    for depth in data['config']['depths']:
        setting=data['tuning']['selected_settings'][str(depth)]
        lines.append(f'At depth {depth}, validation selects {setting["model_shots"]} model shots per point and initial radius {setting["radius"]:.2f}. ')
        if len(set(data['tuning']['scores'][str(depth)]))==1:
            lines.append('All four validation scores are tied exactly; the first candidate in the declared order wins the tie. '
                         'This outcome is not evidence of tuning superiority. ')
        for method in ['hoeffding_iso','bernstein_iso','bernstein_shape']:
            group=followup_group(data,method,depth)
            accepts=sum(decision['decision']=='accepted' for r in group for decision in r['decisions'])
            changed=sum(not np.allclose(r['theta'],r['initial'],atol=1e-12,rtol=0) for r in group)
            total_shots=sum(r['shots'] for r in group)
            step_word='step' if accepts==1 else 'steps'
            lines.append(f'{FOLLOWUP_LABELS[method]} accepts {accepts} {step_word} and changes {changed}/{len(group)} outputs, using {total_shots:,} online shots in total. ')
        iso=graph_means(followup_group(data,'bernstein_iso',depth),'objective')
        shaped=graph_means(followup_group(data,'bernstein_shape',depth),'objective')
        panels={r['graph']:r['panel'] for r in followup_group(data,'bernstein_iso',depth)}
        lines.append(f'The paired expected-cut change from the isotropic to shaped variance-sensitive rule is {fmt_ci(bank_conditional_ci({g:100*(iso[g]-shaped[g]) for g in iso},panels))} percentage points at depth {depth}; positive values favor shaping. ')
    lines += [r'\begin{table}[t]',
              r'\caption{Separate follow-up test comparison. Each deficit is a mean signed reference gap in percentage points with a 95\% graph-bootstrap interval, stratified within the two fixed reference banks (10,000 resamples; seed 5713). Each depth averages 16 graph means, with two shot repetitions per graph. Costs are mean online shots; add 8,192 final-validation shots per output and the separately reported tuning and bank costs.}\label{tab:followup}',
              r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrr}',r'\toprule',
              r'Method & \shortstack{$p=2$ deficit\\{[95\% interval]}} & \shortstack{$p=3$ deficit\\{[95\% interval]}} & \shortstack{$p=2$\\shots} & \shortstack{$p=3$\\shots} \\',r'\midrule']
    for method in data['config']['methods']:
        fields=[FOLLOWUP_LABELS[method]]
        groups=[followup_group(data,method,p) for p in data['config']['depths']]
        for group in groups:
            mean,lo,hi=followup_ci(group)
            fields.append(f'{mean:.2f} [{lo:.2f}, {hi:.2f}]')
        fields.extend(f'{np.mean([r["shots"] for r in group]):,.0f}' for group in groups)
        lines.append(' & '.join(fields)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    lines.append('These graph-level intervals do not estimate uncertainty over a population of reference banks. '
                 'Separate panel means follow; each panel draws its bank, validation split, test split and random streams together, so a difference between panels is not attributable to the reference bank alone: ')
    for panel in sorted({r['panel'] for r in test_runs}):
        for depth in data['config']['depths']:
            groups={method:[r for r in followup_group(data,method,depth) if r['panel']==panel]
                    for method in ['initialization_only','bernstein_iso','bernstein_shape','spsa']}
            means=[np.mean([100*r['signed_reference_gap'] for r in group]) for group in groups.values()]
            lines.append(f'panel {panel+1}, depth {depth}, gives descriptor-anchor, isotropic Bernstein, shaped Bernstein and SPSA deficits of '+
                         ', '.join(f'{mean:.2f}' for mean in means)+' percentage points, respectively. ')
    for depth in data['config']['depths']:
        group=followup_group(data,'initialization_only',depth)
        counts=[sum(r['hitting_shots'][str(gap)] is not None for r in group) for gap in data['config']['target_gaps']]
        lines.append(f'At depth {depth}, the initialization-only anchor meets gaps 0.002, 0.005, 0.01 and 0.02 on '+
                     ', '.join(f'{count}/{len(group)}' for count in counts)+
                     ' test runs, respectively. Repeated initialization-only records correspond to the same anchor within a graph. ')
    if overlap:
        lines.append('An exclusion sensitivity analysis removes the overlapping graph without refitting or rerunning any method. ')
        for depth in data['config']['depths']:
            groups={method:[r for r in followup_group(data,method,depth) if r['graph'] not in overlap]
                    for method in data['config']['methods']}
            means={method:float(np.mean(list(graph_means(group,'signed_reference_gap').values())))*100
                   for method,group in groups.items()}
            ranking=sorted(means,key=means.get)
            iso=graph_means(groups['bernstein_iso'],'objective')
            shaped=graph_means(groups['bernstein_shape'],'objective')
            change=float(np.mean([100*(iso[g]-shaped[g]) for g in iso]))
            lines.append(f'On these {len(tests)-len(overlap)} graphs at depth {depth}, the paired expected-cut change from isotropic to shaped Bernstein is {change:.2f} percentage points; '
                         f'the lowest mean deficits are {FOLLOWUP_LABELS[ranking[0]]} ({means[ranking[0]]:.2f}) and {FOLLOWUP_LABELS[ranking[1]]} ({means[ranking[1]]:.2f}). '
                         f'The shaped rule has deficit {means["bernstein_shape"]:.2f}, compared with {means["initialization_only"]:.2f} for initialization alone. ')
    calls=sum(reference['exact_calls'] for reference in data['references'].values())
    bank_ids={r['id'] for r in data['instances'] if r['split']=='bank'}
    bank_calls=sum(reference['exact_calls'] for key,reference in data['references'].items()
                   if key.rsplit(':p',1)[0] in bank_ids)
    test_shots=sum(r['shots'] for r in test_runs)
    test_validation=sum(r['validation']['shots'] for r in test_runs)
    lines.append(f'The held-out test runs consume {test_shots:,} optimization shots and {test_validation:,} additional validation shots. '
                 f'All follow-up best-found references together use {calls:,} exact objective calls, including {bank_calls:,} bank-reference calls; '
                 'these are classical simulation costs, not uncharged hardware shots. '
                 'The added panels enlarge the tested regime but remain ideal-circuit, small-graph experiments with a fixed, disclosed tuning grid.')
    return lines


def transport_results(data, study):
    rows=data['summary']
    lines=[r'\paragraph{Solver sensitivity limits structural-selection conclusions.}',
           'The separate transport audit repeats every canonical target--reference comparison with longer iterations, three regularizations, multiple starts, and independent conditional-gradient optimization. '
           'All saved couplings are checked by a direct four-index distortion calculation. '
           'Conditional-gradient stopping is checked by a transportation linear-program gap; this is a first-order stationarity diagnostic at the stated numerical tolerance, not a certificate of exact stationarity or global optimality. '
           'Table~\\ref{tab:transport} reports solver failures alongside changed donors and resulting anchor quality. '
           'A target with any infeasible candidate is excluded from that selector\'s quality summary and explicitly counted; no finite score is substituted for an invalid solve.',
           'Quality means over different valid-target subsets are not paired policy comparisons and must not be used to rank those selectors directly.',
           r'\begin{table}[t]',
           r"\caption{Transport sensitivity on 384 canonical graph pairs. ``Unconv.'' and ``Infeas.'' count selected pairwise solver candidates; changed donors and valid targets count graphs out of 24. Depth-specific deficits are mean anchor-only signed reference gaps in percentage points over valid targets. The first two rows are reference points rather than audit solvers: the study's own capped entropic solver and the transport-free descriptor selector, both over all 24 targets. Feasibility, convergence, and useful donor selection are distinct properties.}\label{tab:transport}",
           r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'Solver & Unconv. & Infeas. & \shortstack{Changed\\donors} & \shortstack{Valid\\targets} & \shortstack{$p=1$\\deficit} & \shortstack{$p=2$\\deficit} \\',r'\midrule']
    budget=max(study['config']['budgets'])
    original=[s for key,prior in study['priors'].items() if key.endswith(':p1')
              for s in prior['gw']['solver']]
    def anchor_deficit(method,p):
        group=selected(study,method,p,budget)
        per_graph={g:np.mean([r['initial_objective']-r['reference_objective']
                              for r in group if r['graph']==g])
                   for g in {r['graph'] for r in group}}
        return 100*float(np.mean(list(per_graph.values())))
    for method,label,unconv,infeas in [('gw_iso','Original capped entropic (study)',str(sum(s['residual']>1e-9 for s in original)),'0'),
                                       ('descriptor_iso','Descriptor selector (no transport)','--','--')]:
        lines.append(' & '.join([label,unconv,infeas,'--','24',
                                 f'{anchor_deficit(method,1):.2f}',f'{anchor_deficit(method,2):.2f}'])+r' \\')
    lines.append(r'\midrule')
    for method,label in TRANSPORT_LABELS.items():
        pair=[next(r for r in rows if r['solver']==method and r['depth']==p) for p in [1,2]]
        first=pair[0]
        fields=[label,str(first['unconverged_pairs']),str(first['infeasible_pairs']),str(first['changed_donors']),str(first['valid_targets'])]
        fields.extend('--' if r['mean_deficit_pp'] is None else f'{r["mean_deficit_pp"]:.2f}' for r in pair)
        lines.append(' & '.join(fields)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return lines


REVISION_LABELS={
    'initialization_only':'Initialization only',
    'hoeffding_iso':'Original Hoeffding',
    'revised_hoeffding_iso':'Revised Hoeffding, iso.',
    'revised_hoeffding_shape':'Revised Hoeffding, shaped',
    'revised_bernstein_iso':'Revised Bernstein, iso.',
    'revised_bernstein_shape':'Revised Bernstein, shaped',
    'bernstein_iso':'Original Bernstein, iso.', 'bernstein_shape':'Original Bernstein, shaped',
    'spsa':'SPSA', 'cobyla':'COBYLA'}
REVISION_COLORS={
    'initialization_only':'#777777', 'hoeffding_iso':'#9A7E5C',
    'revised_hoeffding_iso':'#56B4E9','revised_hoeffding_shape':'#CC79A7',
    'revised_bernstein_iso':'#0072B2','revised_bernstein_shape':'#009E73',
    'bernstein_iso':'#6B5CA5','bernstein_shape':'#CC79A7','spsa':'#E69F00','cobyla':'#D55E00'}


def available_objective(run, budget):
    """Use only completed incumbent events, including initialization at zero.

    This is the algorithm's returned point, not a retrospective best objective.
    No interpolation across atomic batches or exact-objective reselection occurs.
    """
    events=run['incumbents']
    if not events or events[0]['shots']!=0:
        raise ValueError('An incumbent trajectory must include initialization at zero shots')
    if any(b['shots']<a['shots'] for a,b in zip(events,events[1:])):
        raise ValueError('Incumbent events must be ordered by completed shot count')
    available=[event for event in events if event['shots']<=budget]
    if not available:
        raise ValueError('Budget must be nonnegative')
    return float(available[-1]['objective'])


def revision_interval(rows, value):
    """Resample paired graph values within fixed-bank panels, averaging seeds."""
    values={g:float(np.mean([value(r) for r in rows if r['graph']==g]))
            for g in sorted({r['graph'] for r in rows})}
    if not values:
        raise ValueError('Cannot summarize an uncomputed comparison')
    return bank_conditional_ci(values,{r['graph']:r['panel'] for r in rows})


def refinement_curves(rows, budgets, target_gap):
    """Equal-graph means of recorded incumbents and cumulative target attainment."""
    graphs=sorted({r['graph'] for r in rows})
    if not graphs:
        raise ValueError('Cannot plot an uncomputed trajectory')
    gains=[]
    successes=[]
    for budget in budgets:
        gains.append(np.mean([np.mean([100*(r['initial_objective']-available_objective(r,budget))
                                      for r in rows if r['graph']==g]) for g in graphs]))
        successes.append(np.mean([np.mean([any(e['shots']<=budget and
                                    e['objective']<=r['reference_objective']+target_gap
                                    for e in r['incumbents'])
                                    for r in rows if r['graph']==g]) for g in graphs]))
    return np.asarray(gains),np.asarray(successes)


def paired_revision_interval(rows, method, comparator, value=None):
    """Paired graph-mean gain (default) or another prespecified run metric."""
    value=value or (lambda r:100*(r['initial_objective']-r['objective']))
    groups={m:{g:np.mean([value(r)
                         for r in rows if r['method']==m and r['graph']==g])
               for g in {r['graph'] for r in rows if r['method']==m}}
            for m in (method,comparator)}
    if groups[method].keys()!=groups[comparator].keys():
        raise ValueError('Paired comparisons require the same target graphs')
    values={g:groups[method][g]-groups[comparator][g] for g in groups[method]}
    return bank_conditional_ci(values,{r['graph']:r['panel'] for r in rows})


def paired_cost_saving_interval(rows, stop, continuation):
    """Percentage reduction in paired mean cost, with graph-stratified resampling."""
    groups={method:graph_means([r for r in rows if r['method']==method],'shots')
            for method in (stop,continuation)}
    if not groups[stop] or groups[stop].keys()!=groups[continuation].keys():
        raise ValueError('Cost fractions require the same nonempty target graphs')
    values={g:[groups[stop][g],groups[continuation][g]] for g in groups[stop]}
    if any(v[1]<=0 for v in values.values()):
        raise ValueError('Continuation costs must be positive')
    result=block_curve_interval(values,{r['graph']:r['panel'] for r in rows},
                                lambda v:100*(v[...,1]-v[...,0])/v[...,1])
    return tuple(float(v) for v in result)


def fresh_prediction_residuals(data):
    """Pair independent empirical/Gaussian frequencies without pooling conditions."""
    lookup={r['id']:r for r in data['predictions']}
    observed_n=data['config']['proposal_repetitions']
    gaussian_n=data['config']['gaussian_repetitions']
    records=[]
    for row in data['schedule_proposals']:
        if row['split']!='fresh' or row['shape_kind']!='shape':
            continue
        probability=lookup[row['prediction_id']]['gaussian_positive_margin_probability']
        observed=row['observed_positive_margin_fraction']
        counts=[observed*observed_n,probability*gaussian_n]
        if any(not np.isclose(v,round(v),atol=1e-10,rtol=0) for v in counts):
            raise ValueError('Saved frequencies must correspond to integer success counts')
        records.append({'graph':row['graph'],'depth':row['depth'],'radius':row['radius'],
                        'error':observed-probability,'observed_successes':round(counts[0]),
                        'gaussian_successes':round(counts[1])})
    return records


def conditional_mc_reference(records, observed_n=32, gaussian_n=256):
    """Exact pointwise conditional null range for an equal-graph frequency difference.

    Under equality of the two Bernoulli probabilities within each graph, condition
    on that graph's total successes. Its observed-sample count is hypergeometric.
    Convolution combines independent graphs, permitting heterogeneous probabilities.
    This is a Monte Carlo equality reference, not a graph-population confidence band.
    """
    from scipy.stats import hypergeom
    if not records or len({r['graph'] for r in records})!=len(records):
        raise ValueError('The conditional reference needs one record per graph')
    if min(observed_n,gaussian_n)<=0:
        raise ValueError('Both repetition counts must be positive')
    pmf=np.array([1.])
    total_successes=0
    for row in records:
        observed=row['observed_successes']
        gaussian=row['gaussian_successes']
        if not (int(observed)==observed and int(gaussian)==gaussian and
                0<=observed<=observed_n and 0<=gaussian<=gaussian_n):
            raise ValueError('Success counts must be integers within the sample sizes')
        successes=int(observed+gaussian)
        total_successes+=successes
        pmf=np.convolve(pmf,hypergeom.pmf(np.arange(observed_n+1),
                                        observed_n+gaussian_n,successes,observed_n))
    pmf/=pmf.sum()
    quantiles=np.searchsorted(np.cumsum(pmf),[.025,.975])
    differences=((1/observed_n+1/gaussian_n)*quantiles-total_successes/gaussian_n)/len(records)
    return tuple(float(v) for v in differences)


def identical_attribution_outputs(rows):
    groups={}
    for row in rows:
        groups.setdefault((row['graph'],row['depth'],row['replicate']),[]).append(row)
    identical=sum(len(group)==4 and all(np.array_equal(group[0]['theta'],r['theta'])
                                        for r in group[1:]) for group in groups.values())
    return identical,len(groups)


def historical_physical_sensitivity(run, shape):
    """Recover physical fitting maps only when periodic displacement is unique.

    Every normalized design point lies in the unit ball. The coordinatewise
    Delta*row_norm(B) bound proves the true displacement is within half a period;
    saved weighted condition numbers independently check reconstructed designs.
    """
    from .search import periods
    shape=np.asarray(shape,dtype=float)
    dimension=len(shape)
    period=periods(dimension//2)
    previous_shots=0
    result=[]
    for decision in run['decisions']:
        radius=decision['radius']
        if np.any(radius*np.linalg.norm(shape,axis=1)>=period/2):
            raise ValueError('Historical model displacements are not uniquely unwrapped by the design bound')
        batch=[row for row in run['evaluations'] if row['stage']=='model' and
               previous_shots<row['cumulative_shots']<=decision['shots']]
        if len(batch)!=dimension+1 or len({row['job'] for row in batch})!=1:
            raise ValueError('Each saved decision must match one complete affine model batch')
        delta=(np.array([row['theta'] for row in batch])-decision['center']+period/2)%period-period/2
        points=np.linalg.solve(shape,delta.T).T/radius
        if np.any(np.linalg.norm(points,axis=1)>1+1e-10):
            raise ValueError('Recovered historical design is outside the normalized unit ball')
        design=np.c_[np.ones(dimension+1),points]
        weighted=design*np.sqrt([row['shots'] for row in batch])[:,None]
        if not np.isclose(np.linalg.cond(weighted),decision['condition'],rtol=1e-8,atol=1e-9):
            raise ValueError('Recovered design does not match the saved weighted condition number')
        coefficient_map=np.linalg.solve(design,np.eye(dimension+1))[1:]
        physical_map=np.linalg.solve((radius*shape).T,coefficient_map)
        result.append(float(np.linalg.norm(physical_map,2)))
        previous_shots=decision['shots']
    return result



def block_curve_interval(values,panels,transform=None):
    """Pointwise graph-block bootstrap; an entire graph trajectory is one unit."""
    keys=sorted(values)
    matrix=np.asarray([values[g] for g in keys],dtype=float)
    rng=np.random.default_rng(5713)
    weights=np.zeros((10000,len(keys)))
    for panel in sorted({panels[g] for g in keys}):
        positions=np.asarray([i for i,g in enumerate(keys) if panels[g]==panel])
        counts=rng.multinomial(len(positions),np.ones(len(positions))/len(positions),size=10000)
        weights[:,positions]=counts
    bootstrap=weights@matrix/len(keys)
    average=matrix.mean(0)
    if transform is not None:
        average=transform(average)
        bootstrap=transform(bootstrap)
    lo,hi=np.quantile(bootstrap,[.025,.975],axis=0)
    return average,lo,hi


def paired_curve_bands(rows,budgets,method,comparator):
    """Compute paired trajectory effects before graph-block resampling."""
    groups={m:{g:[r for r in rows if r['method']==m and r['graph']==g]
               for g in {r['graph'] for r in rows if r['method']==m}} for m in (method,comparator)}
    if groups[method].keys()!=groups[comparator].keys() or not groups[method]:
        raise ValueError('Paired curves require identical nonempty graph sets')
    values={}
    for graph in groups[method]:
        values[graph]=np.array([np.mean([100*(r['initial_objective']-available_objective(r,b))
                                        for r in groups[method][graph]])-
                               np.mean([100*(r['initial_objective']-available_objective(r,b))
                                        for r in groups[comparator][graph]]) for b in budgets])
    return block_curve_interval(values,{r['graph']:r['panel'] for r in rows})


def graph_diagnostic_interval(rows,value,rms=False):
    """Equal graph weights after averaging depth, design and shot replicates."""
    values={g:[np.mean([value(r)**2 if rms else value(r) for r in rows if r['graph']==g])]
            for g in sorted({r['graph'] for r in rows})}
    if not values:
        raise ValueError('Cannot summarize an empty diagnostic condition')
    result=block_curve_interval(values,{g:0 for g in values},np.sqrt if rms else None)
    return tuple(float(v[0]) for v in result)


def _save_figure(fig, target, name, plt):
    fig.savefig(target/name,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def render_transport(data,target,plt,canonical):
    """Compare every audit solver with references on its identical valid targets."""
    methods=list(TRANSPORT_LABELS)
    labels=['Entropic .02','Entropic .05','Entropic .10','Entropic multi','Conditional grad.']
    rows=[next(r for r in data['summary'] if r['solver']==m and r['depth']==2) for m in methods]
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.7))
    x=np.arange(len(methods))
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.15)
    axes[0,0].bar(x-.17,[r['unconverged_pairs'] for r in rows],width=.34,label='Unconverged')
    axes[0,0].bar(x+.17,[r['infeasible_pairs'] for r in rows],width=.34,label='Infeasible')
    axes[0,0].set_xticks(x,labels,rotation=30,ha='right')
    axes[0,0].set_ylabel('Pairwise solves (of 384)')
    axes[0,0].set_title('(a) Distinct solver failures')
    axes[0,0].legend(frameon=False)
    axes[0,1].bar(x,[r['changed_donors'] for r in rows],color='#0072B2')
    axes[0,1].set_xticks(x,[f'{label}\nn = {row["valid_targets"]}' for label,row in zip(labels,rows)],rotation=25,ha='right',fontsize=7)
    axes[0,1].set_ylabel('Changed donors')
    axes[0,1].set_title('(b) Changes among feasible targets')
    for index,row in enumerate(rows):
        axes[0,1].text(index,row['changed_donors']+.3,f"{row['changed_donors']}/{row['valid_targets']}",ha='center',fontsize=7)
    axes[0,1].set_ylim(0,27)
    for depth,ax in zip([1,2],axes[1]):
        for index,method in enumerate(methods):
            row=next(r for r in data['summary'] if r['solver']==method and r['depth']==depth)
            graphs=row['valid_graphs']
            comparisons=[row['graph_deficits_pp']]
            for kind in ['descriptor','gw']:
                comparisons.append([100*(canonical['priors'][f'{g}:p{depth}'][kind]['initial_objective']-
                                         canonical['references'][f'{g}:p{depth}']['objective']) for g in graphs])
            for offset,values,color,label in zip([-.25,0,.25],comparisons,
                                                ['#0072B2','#009E73','#D55E00'],['Audit solver','Descriptor','Original GW']):
                mean,lo,hi=ci(values)
                uncertainty={'yerr':[[mean-lo],[hi-mean]],'capsize':1.3} if len(values)>1 else {}
                ax.bar(index+offset,mean,width=.24,color=color,**uncertainty,
                       label=label if index==0 else None)
        ax.set_xticks(x,[f'{label}\n'+('single target' if row['valid_targets']==1 else f'n = {row["valid_targets"]}')
                        for label,row in zip(labels,rows)],rotation=25,ha='right',fontsize=7)
        ax.axvline(.5,color='.55',lw=.7,ls=':')
        ax.set_ylabel('Anchor reference deficit (pp)')
        ax.set_title(f'({"c" if depth==1 else "d"}) p = {depth}: matched valid-target triples')
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=7)
    fig.suptitle('Single-target quality bars have no interval; comparisons are matched within each solver',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.96))
    _save_figure(fig,target,'transport.pdf',plt)


def corrected_local_deficits(data,depth):
    """Re-evaluate direct-score deterministic selection once per target graph."""
    from .experiment import graph_from
    from .qaoa import MaxCutQAOA
    bank=[r for r in data['instances'] if r['split']=='bank']
    targets={r['id']:r for r in data['instances'] if r['split']=='test'}
    values=[]
    for key,priors in data['priors'].items():
        if not key.endswith(f':p{depth}'):
            continue
        scores=np.asarray(priors['local']['scores'])
        chosen=int(np.flatnonzero(scores<=scores.min()+1e-12)[0])
        theta=data['references'][f'{bank[chosen]["id"]}:p{depth}']['theta']
        graph=key.rsplit(':p',1)[0]
        objective=MaxCutQAOA(graph_from(targets[graph])).objective(theta)
        values.append(100*(objective-data['references'][key]['objective']))
    return values


def render_revision(data,target,plt):
    """Five main figures, all derived from completed revision observations."""
    rows=data['runs']+data['baseline_runs']
    depths=sorted({r['depth'] for r in data['runs']})
    methods=['initialization_only','hoeffding_iso','revised_hoeffding_iso','revised_hoeffding_shape',
             'bernstein_iso','bernstein_shape','revised_bernstein_iso','revised_bernstein_shape','spsa','cobyla']
    budget=max(r['budget'] for r in rows)
    fig,axes=plt.subplots(len(depths),2,figsize=(7.1,7.8),sharey=True,
                          gridspec_kw={'width_ratios':[1,1]})
    y=np.arange(len(methods))
    for j,depth in enumerate(depths):
        left,right=axes[j]
        for index,method in enumerate(methods):
            group=[r for r in rows if r['depth']==depth and r['method']==method]
            initial=revision_interval(group,lambda r:100*(r['initial_objective']-r['reference_objective']))
            final=revision_interval(group,lambda r:100*r['signed_reference_gap'])
            gain=revision_interval(group,lambda r:100*(r['initial_objective']-r['objective']))
            for offset,interval,color,label in [(-.17,initial,'#c8c8c8','Initial'),
                                                (.17,final,REVISION_COLORS[method],'Final')]:
                mean,lo,hi=interval
                left.barh(index+offset,mean,height=.31,color=color,
                          xerr=[[mean-lo],[hi-mean]],capsize=1.5,
                          error_kw={'elinewidth':.7},label=label if index==0 else None)
            mean,lo,hi=gain
            right.barh(index,mean,height=.64,color=REVISION_COLORS[method],
                       xerr=[[mean-lo],[hi-mean]],capsize=2,error_kw={'elinewidth':.8})
            changed=sum(not np.allclose(r['theta'],r['initial'],atol=1e-12,rtol=0) for r in group)
            changed_graphs=len({r['graph'] for r in group if not np.allclose(r['theta'],r['initial'],atol=1e-12,rtol=0)})
            shots=np.mean([r['shots'] for r in group])/1000
            right.text(1.02,index,f'{shots:.1f}k; {changed}/{len(group)}; {changed_graphs}G',
                       va='center',fontsize=7,transform=right.get_yaxis_transform())
        left.set_yticks(y,[REVISION_LABELS[m] for m in methods],fontsize=7)
        for ax in (left,right):
            ax.axvline(0,color='.5',lw=.8)
            ax.grid(axis='x',alpha=.15)
        left.set_title(f'({chr(97+2*j)}) p = {depth}: initial and final')
        right.set_title(f'({chr(98+2*j)}) p = {depth}: refinement gain')
        left.set_xlabel('Signed reference deficit (pp)')
        right.set_xlabel('Improvement from own anchor (pp)')
        right.text(1.02,1.015,'Shots; changed; graphs',fontsize=6.5,transform=right.transAxes)
    axes[0,0].invert_yaxis()
    fig.suptitle('Initial (light gray) and final (row color); graph-bootstrap 95% intervals',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.96))
    _save_figure(fig,target,'refinement.pdf',plt)

    methods_curve=['initialization_only','revised_bernstein_iso','revised_bernstein_shape',
                   'bernstein_iso','bernstein_shape','spsa','cobyla']
    # All unique completion times preserve every atomic-batch transition.
    budgets=np.array(sorted({0,budget}|{e['shots'] for r in rows for e in r['incumbents']}))
    target_gap=min(data.get('config',{}).get('target_gaps',[.002]))
    fig,axes=plt.subplots(3,len(depths),figsize=(7.1,7.0))
    for j,depth in enumerate(depths):
        for method in methods_curve:
            group=[r for r in rows if r['depth']==depth and r['method']==method]
            gain,success=refinement_curves(group,budgets,target_gap)
            for ax,values in [(axes[0,j],gain),(axes[1,j],success)]:
                ax.step(budgets/1000,values,where='post',label=REVISION_LABELS[method],
                        color=REVISION_COLORS[method],lw=1.2,
                        ls='--' if method=='initialization_only' else '-',
                        zorder=5 if method=='initialization_only' else 2)
                ax.set_xlim(0,budget/1000)
                ax.grid(alpha=.15)
        axes[0,j].set_title(f'({chr(97+j)}) p = {depth}: available output')
        axes[1,j].set_title(f'({chr(99+j)}) p = {depth}: target gap {100*target_gap:g} pp')
        axes[1,j].set_xlabel('Cumulative optimization shots (thousands)')
        axes[1,j].set_ylim(-.025,1.025)
        pair=[r for r in rows if r['depth']==depth and r['method'] in ('bernstein_shape','revised_bernstein_shape')]
        mean,lo,hi=paired_curve_bands(pair,budgets,'revised_bernstein_shape','bernstein_shape')
        axes[2,j].step(budgets/1000,mean,where='post',color='#009E73')
        axes[2,j].fill_between(budgets/1000,lo,hi,step='post',color='#009E73',alpha=.18)
        axes[2,j].axhline(0,color='.4',lw=.7)
        axes[2,j].set_title(f'({chr(101+j)}) p = {depth}: revised minus original shaped')
        axes[2,j].set_xlabel('Available optimization budget (thousands of shots)')
        axes[2,j].grid(alpha=.15)
    axes[2,0].set_ylabel('Paired gain difference (pp)')
    axes[0,0].set_ylabel('Refinement gain (pp)')
    axes[1,0].set_ylabel('Fraction ever attaining target')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=7)
    fig.tight_layout(rect=(0,.10,1,1))
    _save_figure(fig,target,'efficiency.pdf',plt)

    render_mechanism(_mechanism_for_report(),target,plt)

    sizes={r['id']:r['n'] for r in data['instances']}
    families={r['id']:r['family'] for r in data['instances']}
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.4))
    for comparator,color in [('spsa','#E69F00'),('cobyla','#D55E00')]:
        for depth,style in zip(depths,['-','--']):
            ns=sorted({sizes[r['graph']] for r in rows})
            intervals=[paired_revision_interval([r for r in rows if r['depth']==depth and sizes[r['graph']]==n],
                                               'revised_bernstein_shape',comparator) for n in ns]
            mean,lo,hi=np.asarray(intervals).T
            axes[0].errorbar(ns,mean,yerr=[mean-lo,hi-mean],fmt='o'+style,ms=3,
                             color=color,capsize=3,label=f'vs {REVISION_LABELS[comparator]}, p = {depth}')
        if comparator!='spsa':
            continue
        for family,style in zip(sorted(set(families[r['graph']] for r in rows)),['-','--',':','-.']):
            intervals=[paired_revision_interval([r for r in rows if r['depth']==depth and families[r['graph']]==family],
                                                'revised_bernstein_shape',comparator) for depth in depths]
            mean,lo,hi=np.asarray(intervals).T
            axes[1].errorbar(depths,mean,yerr=[mean-lo,hi-mean],fmt='o'+style,ms=3,
                             color=color,capsize=3,label=f'vs {REVISION_LABELS[comparator]}, {family.replace("_"," ")}')
    for ax in axes:
        ax.axhline(0,color='.4',lw=.8)
        ax.grid(alpha=.15)
        ax.legend(frameon=False,fontsize=6.2)
    axes[0].set_xticks(ns)
    axes[0].set_xlabel('Number of vertices')
    axes[0].set_ylabel('Shaped Bernstein gain advantage (pp)')
    axes[0].set_title('(a) Paired effects by size')
    axes[1].set_xticks(depths)
    axes[1].set_xlabel('QAOA depth')
    axes[1].set_title('(b) Paired effects by graph family')
    fig.tight_layout()
    _save_figure(fig,target,'generality.pdf',plt)



def replay_mean_interval(records,field,pair_budget):
    """Conservative pointwise bounds for independent, nonidentical replays.

    Cohorts are fixed, proposals have equal weight, and repetition counts must
    agree. Hoeffding applies to this mean without an identical-probability
    assumption. Bounds do not use the empirical zero frequency as zero risk.
    """
    if not records or len({r['repeats'] for r in records})!=1:
        raise ValueError('A fixed cohort with equal replay counts is required')
    scale=pair_budget if field=='mean_acceptance_shots' else 1.
    mean=float(np.mean([r[field] for r in records]))
    epsilon=scale*np.sqrt(np.log(40)/(2*sum(r['repeats'] for r in records)))
    return mean,max(0.,mean-epsilon),min(scale,mean+epsilon)


def fixed_power_cohorts(records):
    """Check identical proposal identities at all budgets before drawing lines."""
    budgets=sorted({r['endpoint_pair_budget'] for r in records})
    groups={budget:{r['proposal_id'] for r in records if r['endpoint_pair_budget']==budget} for budget in budgets}
    if not groups or any(g!=groups[budgets[0]] for g in groups.values()):
        raise ValueError('Every power budget must use the identical fixed proposal cohort')
    return budgets


def _line_interval(ax,x,triples,color,label,style='-',scale=1.):
    mean,lo,hi=np.asarray(triples,dtype=float).T*scale
    ax.plot(x,mean,style,marker='o',ms=3,lw=1.1,color=color,label=label)
    ax.fill_between(x,lo,hi,color=color,alpha=.11)



@lru_cache(maxsize=1)
def _mechanism_for_report():
    """Verify the immutable mechanism bundle once per reporting process."""
    from .mechanism import load_mechanism
    return load_mechanism()


@lru_cache(maxsize=1)
def _decision_for_report():
    from .decision import load_decision
    return load_decision()


def render_decision(data,mechanism,target,plt):
    """Joint proposal/endpoint utility on the same fresh scheduled conditions."""
    radii=sorted({r['radius'] for r in data['joint_records']})
    pair_cap=max(r['endpoint_pair_budget'] for r in data['joint_records'])
    fig,axes=plt.subplots(3,2,figsize=(7.1,7.7))
    for column,depth in enumerate([2,3]):
        ax=axes[0,column]
        predictions=[r for r in mechanism['predictions'] if r['split']=='fresh' and
                     r['depth']==depth and r['shape_kind']=='shape']
        observations=[r for r in mechanism['schedule_proposals'] if r['split']=='fresh' and
                      r['depth']==depth and r['shape_kind']=='shape']
        for records,field,color,label,style in [
                (predictions,'gaussian_positive_margin_probability','#0072B2','Gaussian forecast','--'),
                (observations,'observed_positive_margin_fraction','#D55E00','Sampled observations','-')]:
            intervals=[graph_diagnostic_interval([r for r in records if r['radius']==radius],
                                                 lambda r:r[field]) for radius in radii]
            _line_interval(ax,radii,intervals,color,label,style)
        ax.set_ylim(0,1)
        ax.set_ylabel('Positive-margin probability')
        ax.set_title(f'({"a" if depth==2 else "b"}) Proposal margin, p = {depth}')
        ax.legend(frameon=False,fontsize=6.5,loc='lower left')
        for rule,color in [('hoeffding','#0072B2'),('bernstein','#D55E00')]:
            for population,style,label in [('gaussian','--','forecast'),('sampled','-','observed')]:
                records=[r for r in data['joint_records'] if r['depth']==depth and
                         r['rule']==rule and r['population']==population and r['endpoint_pair_budget']==pair_cap]
                for row,field,scale in [(1,'mean_accepted_true_gain_pp',1),(2,'mean_total_shots',.001)]:
                    intervals=[graph_diagnostic_interval([r for r in records if r['radius']==radius],
                                                         lambda r:r[field]) for radius in radii]
                    _line_interval(axes[row,column],radii,intervals,color,
                                   f'{rule.capitalize()}, {label}',style,scale=scale)
        axes[1,column].axhline(0,color='.4',lw=.7)
        axes[1,column].set_ylabel('Accepted true gain per attempt (pp)')
        axes[1,column].set_title(f'({"c" if depth==2 else "d"}) Accepted utility, p = {depth}')
        axes[2,column].set_ylim(bottom=0)
        axes[2,column].set_ylabel('Model + endpoint shots (thousands)')
        axes[2,column].set_title(f'({"e" if depth==2 else "f"}) Total measured cost, p = {depth}')
    for ax in axes.flat:
        ax.set_xscale('log',base=2)
        ax.set_xticks(radii,[str(v) for v in radii],fontsize=7)
        ax.set_xlabel('Radius along S(radius)')
        ax.grid(alpha=.15)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=7)
    fig.suptitle(f'Fresh shaped conditions; four looks, endpoint-pair cap {pair_cap:,} shots',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.96))
    _save_figure(fig,target,'decision.pdf',plt)


def render_mechanism(data,target,plt):
    """Controlled endpoint budgets, model error, and independent attribution."""
    fig,axes=plt.subplots(2,3,figsize=(7.1,5.3))
    for row,(key,schedule) in enumerate([('online_power','Online: four looks'),('extended_power','Extended: five looks')]):
        records=data[key]
        for rule,color in [('hoeffding','#0072B2'),('bernstein','#D55E00')]:
            for positive,style,name in [(True,'-','h > 0'),(False,'--','h <= 0')]:
                cohort=[r for r in records if r['rule']==rule and (r['acceptance_margin']>0)==positive]
                budgets=fixed_power_cohorts(cohort)
                for col,field in enumerate(['acceptance_probability','unresolved_fraction','mean_acceptance_shots']):
                    intervals=[replay_mean_interval([r for r in cohort if r['endpoint_pair_budget']==b],field,b) for b in budgets]
                    _line_interval(axes[row,col],np.asarray(budgets)/1000,intervals,color,
                                   f'{rule.capitalize()}, {name}',style,scale=.001 if col==2 else 1)
        for col,ax in enumerate(axes[row]):
            ax.set_xscale('log',base=2)
            ticks=sorted({r['endpoint_pair_budget'] for r in records})
            ax.set_xticks(np.array(ticks)/1000,[f'{v/1000:g}' for v in ticks],fontsize=6.5,rotation=25)
            ax.set_xlabel('Endpoint-pair cap (thousands)')
            ax.set_title(f'({chr(97+3*row+col)}) {schedule}',fontsize=8)
            ax.grid(alpha=.15)
            if col<2:
                ax.set_ylim(0,1)
        axes[row,0].set_ylim(0,.06 if row==0 else 1)
        axes[row,0].set_ylabel('Acceptance probability\n(online scale: 0 to 0.06)' if row==0 else
                               'Acceptance probability\n(extended scale: 0 to 1)')
        axes[row,1].set_ylabel('Unresolved fraction')
        axes[row,2].set_ylabel('Actual endpoint shots (thousands)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=7)
    fig.tight_layout(rect=(0,.095,1,1))
    _save_figure(fig,target,'acceptance_power.pdf',plt)

    diagnostics=[r for r in data['model_diagnostics'] if r['split']=='diagnostic' and r['shape_kind']=='shape']
    predictions=[r for r in data['predictions'] if r['split']=='diagnostic' and r['shape_kind']=='shape']
    proposals=[r for r in data['schedule_proposals'] if r['split']=='diagnostic' and r['shape_kind']=='shape']
    radii=sorted({r['radius'] for r in predictions})
    shots=sorted({r['model_shots'] for r in diagnostics})
    fig,axes=plt.subplots(3,2,figsize=(7.1,7.5))
    for radius,color in zip(radii[::2],['#0072B2','#009E73','#D55E00']):
        group=[r for r in diagnostics if r['radius']==radius]
        for ax,field in [(axes[0,0],'rms_sampling_error'),(axes[0,1],'rms_total_error')]:
            intervals=[graph_diagnostic_interval([r for r in group if r['model_shots']==n],lambda r:r[field],rms=True) for n in shots]
            _line_interval(ax,shots,intervals,color,f'Radius {radius:g}')
        bias=graph_diagnostic_interval([r for r in predictions if r['radius']==radius],lambda r:np.sqrt(r['exact_bias_sq']),rms=True)[0]
        axes[0,1].axhline(bias,color=color,ls=':',lw=1)
    for ax in axes[0]:
        ax.set_xscale('log',base=2)
        ax.set_yscale('log')
        ax.set_xticks(shots,[str(n) for n in shots],fontsize=7)
        ax.set_xlabel('Model shots per point')
        ax.set_ylabel('RMS gradient error')
    axes[0,0].set_title('(a) Sampling error around exact fit')
    axes[0,1].set_title('(b) Total error; dotted exact-bias floors')
    axes[0,0].legend(frameon=False,fontsize=7)
    exact=[graph_diagnostic_interval([r for r in predictions if r['radius']==radius],lambda r:float(r['exact_acceptance_margin']>0)) for radius in radii]
    observed=[graph_diagnostic_interval([r for r in proposals if r['radius']==radius],lambda r:r['observed_positive_margin_fraction']) for radius in radii]
    _line_interval(axes[1,0],radii,exact,'#0072B2','Exact-data model','--')
    _line_interval(axes[1,0],radii,observed,'#D55E00','Sampled schedule')
    axes[1,0].set_ylabel('Fraction with positive margin')
    axes[1,0].set_ylim(-.03,1.03)
    axes[1,0].set_title('(c) Proposal quality along S(radius)')
    axes[1,0].legend(frameon=False,fontsize=7,loc='lower left')
    exact_h=[graph_diagnostic_interval([r for r in predictions if r['radius']==radius],lambda r:100*r['exact_acceptance_margin']) for radius in radii]
    observed_h=[graph_diagnostic_interval([r for r in proposals if r['radius']==radius],lambda r:100*np.mean([q['acceptance_margin'] for q in r['proposals']])) for radius in radii]
    _line_interval(axes[1,1],radii,exact_h,'#0072B2','Exact-data model','--')
    _line_interval(axes[1,1],radii,observed_h,'#D55E00','Sampled schedule')
    axes[1,1].axhline(0,color='.5',lw=.7)
    axes[1,1].set_yscale('symlog',linthresh=.1)
    axes[1,1].set_ylabel('Mean true margin (pp; symmetric log)')
    axes[1,1].set_title('(d) Margin magnitude along S(radius)')
    for depth,color in [(2,'#0072B2'),(3,'#D55E00')]:
        cost=[np.mean([(2*depth+1)*r['scheduled_shots'] for r in predictions if r['depth']==depth and r['radius']==radius]) for radius in radii]
        axes[2,0].plot(radii,cost,'o-',color=color,ms=3,label=f'p = {depth}: model cost')
    allocation=[next(r['scheduled_shots'] for r in predictions if r['radius']==radius) for radius in radii]
    axes[2,0].plot(radii,allocation,'o--',color='#555555',ms=3,label='Shots per point')
    axes[2,0].set_yscale('log',base=2)
    axes[2,0].set_ylabel('Scheduled model shots')
    axes[2,0].set_title('(e) Actual radius-dependent allocation')
    axes[2,0].legend(frameon=False,fontsize=6.5)
    scheduled=[r for r in diagnostics if r['is_schedule']]
    for field,color,label in [('rms_sampling_error','#0072B2','Observed sampling'),('rms_total_error','#D55E00','Observed total')]:
        intervals=[graph_diagnostic_interval([r for r in scheduled if r['radius']==radius],lambda r:r[field],rms=True) for radius in radii]
        _line_interval(axes[2,1],radii,intervals,color,label)
    predicted=[graph_diagnostic_interval([r for r in scheduled if r['radius']==radius],lambda r:np.sqrt(r['predicted_total_mse']),rms=True) for radius in radii]
    _line_interval(axes[2,1],radii,predicted,'#555555','Predicted total','--')
    axes[2,1].set_ylabel('RMS gradient error')
    axes[2,1].set_title('(f) Accuracy along the implemented schedule')
    axes[2,1].legend(frameon=False,fontsize=6.5)
    for ax in axes[1:].flat:
        ax.set_xscale('log',base=2)
        ax.set_xticks(radii,[str(r) for r in radii],fontsize=7)
        ax.set_xlabel('Trust-region radius')
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle('Shaped models: eight diagnostic graphs, depths 2 and 3 equally averaged',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.96))
    _save_figure(fig,target,'components.pdf',plt)

    render_attribution(data,target,plt)
    if (CODE_ROOT/'results'/'decision.json.gz').is_file():
        render_decision(_decision_for_report(),data,target,plt)


def render_attribution(data,target,plt):
    """Draw controlled effects and calibration with a shared external legend."""
    radii=sorted({r['radius'] for r in data['predictions']})
    fig,axes=plt.subplots(3,2,figsize=(7.1,7.5))
    methods=['fixed_continue','fixed_stop','radius_continue','radius_stop']
    for depth,color,offset in [(2,'#0072B2',-.17),(3,'#D55E00',.17)]:
        for index,method in enumerate(methods):
            group=[r for r in data['attribution_runs'] if r['depth']==depth and r['method']==method]
            for ax,value in [(axes[0,0],lambda r:100*(r['initial_objective']-r['objective'])),
                             (axes[0,1],lambda r:r['shots']/1000)]:
                mean,lo,hi=revision_interval(group,value)
                ax.bar(index+offset,mean,width=.34,color=color,yerr=[[mean-lo],[hi-mean]],capsize=2,
                       label=f'p = {depth}' if index==0 else None)
    for ax in axes[0]:
        ax.set_xticks(range(4),['Fixed\ncontinue','Fixed\nstop','Radius\ncontinue','Radius\nstop'],fontsize=7)
        ax.axhline(0,color='.5',lw=.7)
    axes[0,1].legend(frameon=False)
    axes[0,0].set_ylabel('Paired refinement gain (pp)')
    axes[0,1].set_ylabel('Actual optimization shots (thousands)')
    axes[0,0].set_title('(a) Sampling x unresolved-policy attribution')
    axes[0,1].set_title('(b) Cost of the same controlled policies')
    identical,total=identical_attribution_outputs(data['attribution_runs'])
    axes[0,0].set_ylim(top=1.2*axes[0,0].get_ylim()[1])
    axes[0,0].text(.02,.98,f'Identical output vectors: {identical}/{total} matched cases',
                   va='top',fontsize=6.8,transform=axes[0,0].transAxes)
    for depth,color,offset in [(2,'#0072B2',-.17),(3,'#D55E00',.17)]:
        group=[r for r in data['attribution_runs'] if r['depth']==depth]
        contrasts=[[(f'{a}_stop',f'{a}_continue') for a in ('fixed','radius')],
                   [(f'radius_{p}',f'fixed_{p}') for p in ('stop','continue')]]
        for ax,pairs in zip(axes[1],contrasts):
            for index,(method,baseline) in enumerate(pairs):
                mean,lo,hi=paired_revision_interval(group,method,baseline,lambda r:r['shots']/1000)
                ax.bar(index+offset,mean,width=.34,color=color,yerr=[[mean-lo],[hi-mean]],
                       capsize=2,label=f'p = {depth}' if index==0 else None)
    for ax,labels,title in zip(axes[1],[['Fixed allocation','Radius allocation'],['Stop policy','Continue policy']],
                                ['(c) Stopping minus continuation','(d) Radius minus fixed allocation']):
        ax.set_xticks(range(2),labels,fontsize=7)
        ax.axhline(0,color='.4',lw=.7)
        ax.set_ylabel('Paired shot difference (thousands)')
        ax.set_title(title)
    residuals=fresh_prediction_residuals(data)
    for column,depth in enumerate([2,3]):
        ax=axes[2,column]
        groups=[[r for r in residuals if r['depth']==depth and r['radius']==radius] for radius in radii]
        null=np.array([conditional_mc_reference(group,data['config']['proposal_repetitions'],
                                                data['config']['gaussian_repetitions']) for group in groups])*100
        ax.fill_between(radii,null[:,0],null[:,1],color='.75',alpha=.45,
                        label='95% conditional MC null reference')
        intervals=[graph_diagnostic_interval(group,lambda r:100*r['error']) for group in groups]
        mean,lo,hi=np.array(intervals).T
        ax.errorbar(radii,mean,yerr=[mean-lo,hi-mean],color='#D55E00',fmt='o-',ms=3,
                    capsize=2,lw=1,label='Signed mean; graph-bootstrap interval')
        ax.axhline(0,color='.4',lw=.7)
        ax.set_ylabel('Observed minus Gaussian (pp)')
        ax.set_title(f'({"e" if depth==2 else "f"}) Fresh signed calibration, p = {depth}')
    for ax in axes[2:].flat:
        ax.set_xscale('log',base=2)
        ax.set_xticks(radii,[str(r) for r in radii],fontsize=7)
        ax.set_xlabel('Radius along S(radius)')
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.15)
    handles,labels=axes[2,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=6.5)
    fig.tight_layout(rect=(0,.05,1,1))
    _save_figure(fig,target,'attribution.pdf',plt)


def render(data, target):
    # Keep rendering caches in the workspace only. These are removed on success.
    import os
    import shutil
    cache=CODE_ROOT/'results'/'.render-cache'
    os.environ['MPLCONFIGDIR']=str(cache)
    os.environ['XDG_CACHE_HOME']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                         'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.bbox':'tight'})
    target.mkdir(parents=True,exist_ok=True)
    methods=['uniform_iso','random_iso','descriptor_iso','descriptor_shape',
             'gw_iso','gw_shape','local_iso','cobyla','spsa']
    budget=max(data['config']['budgets'])
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.5))
    for column,p in enumerate([1,2]):
        groups=[('Descriptor anchor','descriptor_iso',True),('GW anchor','gw_iso',True)]
        groups.extend((LABELS[m],m,False) for m in methods)
        competitive=[('Descriptor anchor','descriptor_iso',True),('GW anchor','gw_iso',True),
                     ('Descriptor, shaped','descriptor_shape',False),('GW, shaped','gw_shape',False),
                     ('COBYLA','cobyla',False),('SPSA','spsa',False)]
        for row,selection in enumerate([groups,competitive]):
            ax=axes[row,column]
            for j,(label,method,initial) in enumerate(selection):
                runs=selected(data,method,p,budget)
                field=lambda r:100*((r['initial_objective'] if initial else r['objective'])-r['reference_objective'])
                values=[np.mean([field(r) for r in runs if r['graph']==g]) for g in sorted({r['graph'] for r in runs})]
                mean,lo,hi=ci(values)
                ax.barh(j,mean,xerr=[[mean-lo],[hi-mean]],height=.65,
                        color='#999999' if initial else COLORS[j%len(COLORS)],capsize=2,error_kw={'elinewidth':.8})
            ax.set_yticks(range(len(selection)),[v[0] for v in selection],fontsize=7)
            ax.invert_yaxis()
            ax.axvline(0,color='.5',lw=.7)
            ax.axvline(2,color='.5',lw=.8,ls=':',label='Target: 2 pp')
            ax.set_title(f'p = {p}: '+('selected historical methods' if row==0 else 'competitive methods'))
            ax.set_xlabel('Signed reference deficit (pp)')
            ax.grid(axis='x',alpha=.15)
            if row==1:
                ax.set_xlim(min(-.05,ax.get_xlim()[0]),2.15)
        axes[0,column].legend(frameon=False,fontsize=7,loc='lower right')
    fig.tight_layout()
    _save_figure(fig,target,'quality.pdf',plt)

    fig,axes=plt.subplots(1,2,figsize=(7.1,3.3),sharey=True)
    kinds=['uniform','descriptor','gw','local','corrected_local','shuffled']
    colors=['#526777','#0072B2','#D55E00','#CC79A7','#009E73','#999999']
    for ax,p in zip(axes,[1,2]):
        for j,(kind,color) in enumerate(zip(kinds,colors)):
            values=(corrected_local_deficits(data,p) if kind=='corrected_local' else
                    [100*(priors[kind]['initial_objective']-data['references'][key]['objective'])
                     for key,priors in data['priors'].items() if key.endswith(f':p{p}')])
            mean,lo,hi=ci(values)
            ax.barh(j,mean,xerr=[[mean-lo],[hi-mean]],height=.65,color=color,
                    capsize=2,error_kw={'elinewidth':.8})
        ax.set_yticks(range(6),['Fixed donor','Descriptor','GW','Historical Local TV','Corrected Local TV','Shuffled'],fontsize=7)
        ax.set_title(f'Depth p = {p}; 24 deterministic anchors')
        ax.axvline(2,color='.5',ls=':',lw=.8)
        ax.set_xlabel('Initial reference deficit (pp)')
        ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis()
    fig.tight_layout()
    _save_figure(fig,target,'transfer.pdf',plt)

    # Separate canonical and follow-up validation schedules explicitly.
    fig,axes=plt.subplots(2,2,figsize=(7.1,7.8),gridspec_kw={'height_ratios':[1,1.5]})
    methods2=['descriptor_iso','descriptor_shape','gw_shape','no_repair','fixed_radius','fixed_shots']
    from .followup import load_refinement
    revision=load_refinement()
    for column,p in enumerate([1,2]):
        historical=[(LABELS[m],selected(data,m,p,budget)) for m in methods2]
        depth=p+1
        matched=[(REVISION_LABELS[m],[r for r in revision['runs']+revision['baseline_runs'] if r['depth']==depth and r['method']==m])
                 for m in ['initialization_only','bernstein_shape','revised_bernstein_shape','spsa','cobyla']]
        for method,label in [('fixed_continue','Control: fixed / continue'),('fixed_stop','Control: fixed / stop'),
                             ('radius_continue','Control: radius / continue'),('radius_stop','Control: radius / stop')]:
            matched.append((label,[r for r in _mechanism_for_report()['attribution_runs']
                                   if r['depth']==depth and r['method']==method]))
        for row,selection in enumerate([historical,matched]):
            ax=axes[row,column]
            for j,(label,runs) in enumerate(selection):
                left=0.
                for stage,color in [('model','#0072B2'),('acceptance','#E69F00'),('other','#009E73'),('validation','#777777')]:
                    values=[r['validation']['shots'] if stage=='validation' else
                            sum(e['shots'] for e in r['evaluations'] if (e['stage'] not in ('model','acceptance') if stage=='other' else e['stage']==stage)) for r in runs]
                    width=float(np.mean(values))/1000
                    stage_label={'model':'Model fitting','acceptance':'Endpoint testing',
                                 'other':'Baseline objective evaluation','validation':'Final evaluation'}[stage]
                    ax.barh(j,width,left=left,color=color,label=stage_label if j==0 else None)
                    left+=width
            ax.set_yticks(range(len(selection)),[v[0] for v in selection],fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel('Mean measured shots (thousands)')
            ax.set_title(f'Canonical p = {p}: evaluation 16,384' if row==0 else f'Matched p = {depth}: evaluation 8,192')
            ax.grid(axis='x',alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.05,1,1))
    _save_figure(fig,target,'accounting.pdf',plt)

    fig,axes=plt.subplots(1,3,figsize=(7.1,3.2))
    for method,color in [('descriptor_shape','#0072B2'),('no_repair','#D55E00')]:
        rows=selected(data,method,2,max(data['config']['budgets']))
        cond=sorted(d['condition'] for r in rows for d in r['decisions'])
        axes[0].step(cond,np.arange(1,len(cond)+1)/len(cond),where='post',label=LABELS[method],color=color)
        sensitivity=sorted(value for run in rows for value in historical_physical_sensitivity(
            run,data['priors'][f'{run["graph"]}:p2']['descriptor']['shape']))
        axes[1].step(sensitivity,np.arange(1,len(sensitivity)+1)/len(sensitivity),where='post',
                     label=f'{LABELS[method]} ({len(sensitivity)} trials)',color=color)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Weighted design condition number')
    axes[0].set_ylabel('Fraction of recorded trial designs')
    axes[0].set_title('(a) Historical trials, p = 2',fontsize=8)
    axes[0].text(.02,.05,f'Cap {budget:,}; equal trial weights',transform=axes[0].transAxes,fontsize=6)
    axes[0].legend(frameon=False,fontsize=6)
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'Physical fitting-map $\|L\|_2$ (rad$^{-1}$)')
    axes[1].set_ylabel('Fraction of recorded trial designs')
    axes[1].set_title('(b) Same historical trial population',fontsize=8)
    axes[1].legend(frameon=False,fontsize=5.6,loc='lower right')
    for p,color in [(1,'#0072B2'),(2,'#D55E00')]:
        values=[x for key,priors in data['priors'].items() if key.endswith(f':p{p}') for x in priors['local']['scores']]
        counts=[sum(not np.isclose(v,1.) for v in values),sum(np.isclose(v,1.) for v in values)]
        bars=axes[2].bar(np.arange(2)+(p-1.5)*.34,counts,width=.34,color=color,label=f'Depth {p}')
        axes[2].bar_label(bars,padding=2,fontsize=7)
    axes[2].set_xticks([0,1],['Not saturated\n(< 1)','Saturated\n(= 1)'],fontsize=7)
    axes[2].set_ylim(0,440)
    axes[2].set_xlabel('Local-neighborhood certificate')
    axes[2].set_ylabel('Target/reference pairs')
    axes[2].set_title('(c) Certificate saturation',fontsize=8)
    axes[2].legend(frameon=False,fontsize=6)
    fig.tight_layout()
    fig.savefig(target/'conditioning.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    transport_path=CODE_ROOT/'results'/'transport.json.gz'
    if transport_path.is_file():
        with gzip.open(transport_path,'rt') as stream:
            render_transport(json.load(stream),target,plt,data)
    revision_path=CODE_ROOT/'results'/'refinement.json.gz'
    if revision_path.is_file():
        from .followup import load_refinement
        render_revision(load_refinement(revision_path),target,plt)
    shutil.rmtree(cache)


def write_results(data):
    budget=max(data['config']['budgets'])
    c=data['config']
    rows=data['runs']
    initial_counts={p:sum(r['initial_objective']<=r['reference_objective']+c['target_gap'] for r in selected(data,'descriptor_iso',p,budget)) for p in [1,2]}
    initial_graphs={p:len({r['graph'] for r in selected(data,'descriptor_iso',p,budget)
                           if r['initial_objective']<=r['reference_objective']+c['target_gap']})
                    for p in [1,2]}
    total_graphs=len({r['graph'] for r in selected(data,'descriptor_iso',2,budget)})
    total_per=len(selected(data,'descriptor_iso',2,budget))
    benefit=paired(data,'random_iso','descriptor_iso',2,budget)
    fixed=paired(data,'uniform_iso','descriptor_iso',2,budget)
    shape=paired(data,'descriptor_iso','descriptor_shape',2,budget)
    gw=paired(data,'descriptor_iso','gw_shape',2,budget)
    refcalls=sum(r['exact_calls'] for key,r in data['references'].items() if key.startswith('bank_'))
    refseconds=sum(r['seconds'] for key,r in data['references'].items() if key.startswith('bank_'))
    gwseconds=np.mean([prior['gw']['structural_seconds'] for key,prior in data['priors'].items() if key.endswith(':p1')])
    descriptorseconds=np.mean([prior['descriptor']['seconds'] for key,prior in data['priors'].items() if key.endswith(':p1')])
    accepted=sum(d['decision']=='accepted' for r in rows for d in r['decisions'])
    unresolved=sum(d['decision']=='unresolved' for r in rows for d in r['decisions'])
    decisions=sum(len(r['decisions']) for r in rows)
    donor_methods=['uniform_iso','descriptor_iso','descriptor_shape','gw_iso','gw_shape',
                   'local_iso','no_repair','fixed_radius','fixed_shots']
    donor_accepts=sum(d['decision']=='accepted' for r in rows if r['method'] in donor_methods for d in r['decisions'])
    donor_runs=[r for r in rows if r['method'] in donor_methods]
    donor_shots=sum(r['shots'] for r in donor_runs)
    margins=acceptance_margin_diagnostics(data)
    adaptive_margins=margins['counts']['adaptive_donor']
    power_mantissa,power_exponent=f'{margins["adaptive_per_trial_acceptance_bound"]:.2e}'.split('e')
    solvers=[solver for key,prior in data['priors'].items() if key.endswith(':p1')
             for solver in prior['gw']['solver']]
    unfinished=sum(solver['residual']>1e-9 for solver in solvers)
    if donor_accepts:
        raise ValueError('The manuscript interpretation requires zero donor-policy refinements; review the changed findings before reporting.')
    deficits={method:float(np.mean([100*r['signed_reference_gap'] for r in selected(data,method,2,budget)]))
              for method in ['spsa','descriptor_iso','gw_shape']}
    text=[r'\paragraph{The comparison separates initialization from refinement.}',
          f'The study comprises {len(rows):,} optimization runs on 24 held-out graphs, using two depths, two shot caps, three shot seeds, and 14 methods. '
          'All graphs, settings, call records, donor fits, and final validation measurements are saved. '
          'The reference bank contains 16 nonisomorphic graphs; held-out graphs are also nonisomorphic to every bank graph. '
          'The same graph instances and resource caps are used across all policies in this ideal-circuit simulation study.',
          r'\paragraph{Initialization and region shape have distinct effects.}',
          f'At depth 2 and the larger shot cap, descriptor transfer with an isotropic region changes the mean normalized expected cut relative to the fixed-donor control by {fmt_ci(fixed)} percentage points, and relative to a random start by {fmt_ci(benefit)} percentage points. '
          f'Keeping the descriptor initialization fixed, the shaped region changes it by {fmt_ci(shape)} percentage points. '
          f'GW with a shaped region changes it relative to descriptor transfer with an isotropic region by {fmt_ci(gw)} percentage points. '
          'Positive values indicate higher expected cut for the method being compared against its stated baseline. '
          'Intervals resample the 24 graph-level means after averaging the three shot seeds; they are descriptive, unadjusted percentile bootstrap intervals, not simultaneous confidence statements. '
          'The full ablation table prevents an initialization benefit from being attributed to anisotropy (Table~\\ref{tab:results}).',
          f'The nine undisplaced donor-based local-model policies accept {donor_accepts} refinement steps in {len(donor_runs):,} runs across both depths and budgets, despite consuming {donor_shots:,} online shots. '
          'For these policies, the final parameter vector equals the transferred anchor: their quality comes entirely from initialization. '
          f"An explicit initialization-only comparison therefore gives the same depth-2 deficits, {deficits['descriptor_iso']:.2f} percentage points for the descriptor anchor and {deficits['gw_shape']:.2f} for the GW anchor, at zero online refinement shots. "
          'This comparison retains the offline donor-construction cost and does not waive any independent output-validation measurements. '
          f'The {accepted:,} accepted moves occur only in the random-start, displaced-prior, and shuffled-reference controls. '
          f"At the larger depth-2 budget, SPSA has a mean reference deficit of {deficits['spsa']:.2f} percentage points, compared with {deficits['descriptor_iso']:.2f} for descriptor-based local-model search and {deficits['gw_shape']:.2f} for GW-based local-model search. "
          'The prescribed conservative refinement therefore has no demonstrated efficiency advantage over directly using the descriptor-selected anchor.',
          f'The descriptor initialization already meets the prespecified target on {initial_graphs[1]}/{total_graphs} graphs at depth 1 and {initial_graphs[2]}/{total_graphs} graphs at depth 2, before target-graph measurements. '
          f'The corresponding run counts are {initial_counts[1]}/{total_per} and {initial_counts[2]}/{total_per}, because each graph contributes three identical initializations; repeated shot seeds add no independent graphs. '
          'Zero target-graph shots therefore do not mean zero total cost: donor optimization is performed offline. '
          'The target is an expected normalized cut within 0.02 of an independently computed best-found depth-matched reference; it is not a certified QAOA optimum. '
          'A negative deficit means that a run exceeds that reference.',
          r'\paragraph{Conservative decisions consume the available information.}',
          f'Across all trust-region variants, {accepted:,} of {decisions:,} attempted decisions are accepted and {unresolved:,} remain unresolved at their sampling cap or available budget. '
          'Unresolved decisions keep the current point. '
          'Figure~\\ref{fig:accounting} separates fitting from acceptance measurements. '
          'Fixed-shot decisions start at the largest planned sample count; adaptive decisions may stop at an earlier certified look. '
          f"That single look spends {2*c['optimizer']['acceptance_looks'][-1]:,} shots on the two endpoints, more than the smaller {min(c['budgets']):,}-shot cap, so the fixed-shot policy can take no acceptance measurement at that cap; {sum(d['decision']=='unresolved' and d['acceptance_shots_per_point']==0 for r in rows for d in r['decisions']):,} unresolved decisions are recorded with zero acceptance shots. "
          'Their confidence penalties differ because the fixed-shot variant has one planned look. '
          'These are finite-budget comparisons of complete policies, not a demonstration of asymptotic convergence.',
          f'\nAn offline evaluation of the recorded endpoints finds {adaptive_margins["positive_true_decrease"]:,} truly improving proposals among {adaptive_margins["trials"]:,} trials of the eight adaptive undisplaced donor policies. '
          f'Of those trials, {adaptive_margins["margin_at_most_threshold"]:,} ({100*adaptive_margins["margin_at_most_threshold"]/adaptive_margins["trials"]:.1f}\\%) have true acceptance margin $h=f(x)-f(x^+)-\\eta q\\leq0.005$; positive decreases are counted above a $10^{{-12}}$ numerical tolerance. '
          f'For this margin range, Proposition~\\ref{{prop:acceptancepower}} bounds the conditional probability of acceptance in any one trial by ${power_mantissa}\\times10^{{{int(power_exponent)}}}$, even if all four prescribed looks are affordable. '
          'The fixed-shot control is excluded from this calculation because it uses a different confidence penalty. '
          'These exact proposal values are retrospective diagnostics, never online inputs. '
          'The counts include dependent trials and are not independent observations or an empirical estimate of the acceptance probability; they connect the transferred-anchor experiment to the regime in which the proposition predicts low acceptance probability.',
          f'\nBuilding the reference bank used {refcalls:,} exact objective evaluations and {refseconds:.1f} seconds in the recorded environment. '
          'Those exact simulator calls have no assigned hardware-shot equivalent and must not be omitted from deployment considerations. '
          'Every final point receives 16,384 additional independent validation shots, one evaluation, and one simulated job. '
          'Tables show optimization cost separately so that the common validation cost can be added explicitly. '
          'Hardware-job counts describe the specified batching schedule; no device latency or hardware run is inferred.',
          f'The structural transport comparisons require {gwseconds:.2f} seconds per target graph on average, shared between the two depth-specific banks. '
          f'Descriptor selection and shape construction require {1000*descriptorseconds:.2f} milliseconds at depth one. '
          'This substantial classical overhead provides an additional reason to prefer the simpler selector for this study; it is not converted into a hardware speedup.',
          f'Of the {len(solvers)} distinct target--reference transport solves, {unfinished} end above the prescribed $10^{{-9}}$ coupling-change tolerance after the iteration cap. '
          'Their couplings satisfy the marginal constraints, but feasibility does not establish convergence or global optimality. '
          'These finite-iteration scores must be interpreted together with solver and donor-sensitivity diagnostics rather than as exact graph distances.',
          r'\begin{table}[t]',r'\caption{Complete depth-2 comparison at an optimization cap of 131,072 shots. Deficit is 100 times the signed difference from the best-found normalized reference objective. Each row averages 24 graphs and three shot seeds. Target success counts all 72 runs, including zero-shot initial successes. Capped target cost assigns the full budget to failures and uses the first successful recorded incumbent for successes. The first two rows are initialization-only policies that return the selected anchor and make no optimization call, so refinement gains are never credited to transfer. Shots exclude 16,384 independent final-validation shots per run, which every policy including initialization alone still incurs. The Local TV and Fixed donor rows differ only through a floating-point tie-break among depth-2 scores that are all equal to one; see the text.}\label{tab:results}',
          r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
          r'Method & Deficit & Success & Shots & Calls & Jobs & \shortstack{Capped\\target cost} \\',r'\midrule']
    for method,label in [('descriptor_iso','Descriptor anchor only'),('gw_shape','GW anchor only')]:
        group=selected(data,method,2,budget)
        hit=[r['initial_objective']<=r['reference_objective']+c['target_gap'] for r in group]
        anchor_gap=np.mean([100*(r['initial_objective']-r['reference_objective']) for r in group])
        text.append(f"{label} & {anchor_gap:.2f} & {sum(hit)}/72 & 0 & 0.0 & 0.0 & {np.mean([0 if h else budget for h in hit]):.0f}" + r" \\")
    text.append(r'\midrule')
    for method in c['methods']:
        group=selected(data,method,2,budget)
        gap=np.mean([100*r['signed_reference_gap'] for r in group])
        success=sum(r['hitting_shots'] is not None for r in group)
        targetcost=np.mean([r['hitting_shots'] if r['hitting_shots'] is not None else budget for r in group])
        text.append(f"{LABELS[method]} & {gap:.2f} & {success}/72 & {np.mean([r['shots'] for r in group]):.0f} & {np.mean([r['calls'] for r in group]):.1f} & {np.mean([r['jobs'] for r in group]):.1f} & {targetcost:.0f}" + r" \\")
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table}',
             r'\paragraph{Model geometry and the graph bound have limited roles.}']
    for method in ['descriptor_shape','no_repair']:
        conditions=[d['condition'] for r in selected(data,method,2,budget) for d in r['decisions']]
        text.append(f"For {LABELS[method].lower()}, the median local-design condition number is {np.median(conditions):.2f}, with 95th percentile {np.quantile(conditions,.95):.2f}. ")
        sensitivities=[v for run in selected(data,method,2,budget) for v in historical_physical_sensitivity(
            run,data['priors'][f'{run["graph"]}:p2']['descriptor']['shape'])]
        text.append(f'On the same {len(sensitivities)} recorded trial designs, the physical fitting-map '
                    f'operator norm has median {np.median(sensitivities):.2f} and 95th percentile '
                    f'{np.quantile(sensitivities,.95):.2f} in inverse radians. ')
    text.append('These two empirical distributions weight trials equally at depth two and the 131,072-shot cap; '
                'runs with more trials contribute more points. Physical maps are recovered from the recorded '
                'model locations and centers only after verifying that every $\\Delta\\|B_{j,:}\\|_2$ '
                'is less than half its parameter period. This bound makes the periodic displacement unique; '
                'reconstructed normalized designs also reproduce the saved weighted condition numbers. '
                'The physical map is $L=(\\Delta B)^{-\\mathsf T}(A^{-1})_{2:d+1,:}$. '
                'Improved conditioning or sensitivity does not by itself demonstrate useful proposals or saved shots.')
    for p in [1,2]:
        values=[x for key,prior in data['priors'].items() if key.endswith(f':p{p}') for x in prior['local']['scores']]
        text.append(f'At depth {p}, {sum(np.isclose(v,1) for v in values)}/{len(values)} target--reference pairs have local-neighborhood total variation equal to one. ')
    depth2=[prior for key,prior in data['priors'].items() if key.endswith(':p2')]
    spread=max(max(prior['local']['scores'])-min(prior['local']['scores']) for prior in depth2)
    exponent=int(np.ceil(np.log10(spread)))
    tied=sum(prior['local']['anchor']!=prior['uniform']['anchor'] for prior in depth2)
    text.append(f'At depth 2 those recorded scores are constant across the bank to within $10^{{{exponent}}}$, which is double-precision rounding in the edge-frequency sums and not structural information. '
                f'The softmax floor $\\tau=10^{{-3}}$ in the selection rule nevertheless resolves that rounding, so the local-total-variation anchor differs from the fixed-donor anchor on {tied} of {len(depth2)} depth-2 targets, that is {tied*c["repetitions"]} of {total_per} runs per budget. '
                'The separation between the Local TV and Fixed donor rows of Table~\\ref{tab:results} is an artifact of that tie-break and must not be read as evidence of structural selection. ')
    text.append('A bound of one cannot narrow the unit range of the normalized objective. '
                'Improved numerical conditioning alone is not evidence of saved shots, and a valid transfer bound can be uninformative on the evaluated graphs (Fig.~\\ref{fig:conditioning}). '
                'The results do not establish a consistent improvement from the complete transport-shaped policy over the simpler descriptor and conventional-optimizer controls. '
                'Comparisons with a tuned ASTRO-DF implementation and physical device noise remain outside this study.')
    followup_path=CODE_ROOT/'results'/'followup.json.gz'
    transport_path=CODE_ROOT/'results'/'transport.json.gz'
    if followup_path.is_file():
        from .followup import load_followup
        text.extend(followup_results(load_followup(),data))
    if transport_path.is_file():
        from .transport_audit import verify
        verify(transport_path)
        with gzip.open(transport_path,'rt') as stream:
            text.extend(transport_results(json.load(stream),data))
    (REPO_ROOT/'submission'/'tables'/'results.tex').write_text('\n'.join(line.rstrip() for line in text)+'\n')
    if (CODE_ROOT/'results'/'refinement.json.gz').is_file():
        from .followup import load_refinement
        write_refinement_results(load_refinement())



def write_refinement_results(data):
    """Generate revision claims from saved records, keeping historical failures."""
    rows=data['runs']+data['baseline_runs']
    depths=sorted({r['depth'] for r in data['runs']})
    graph_ids={r['graph'] for r in data['runs']}
    shots=sum(r['shots'] for r in data['runs'])
    validation=sum(r['validation']['shots'] for r in data['runs'])
    lines=[r'\paragraph{The matched shape-by-rule experiment separates transfer from refinement.}',
           f'The exploratory matched optimizer experiment evaluates {len(data["runs"])} runs on {len(graph_ids)} '
           'previously studied target graphs, with byte-identical graph-specific descriptor anchors and the '
           'previously selected initial radius and model-shot count. '
           'The full isotropic/shaped by Hoeffding/empirical-Bernstein factorial changes the '
           'radius-dependent sampling and stopping logic without retuning on those targets. '
           f'Its measured cost is {shots:,} optimization shots plus {validation:,} independent '
           'output-validation shots. The historical SPSA, COBYLA and original Bernstein runs '
           'provide matched-anchor comparators at the same cap. The original Bernstein rule '
           'is our own implementation; it is not an implementation of a published variance-aware optimizer. '
           'This post-hoc comparison is exploratory because these graphs were already used to diagnose the original method.',
           r'\paragraph{Returned quality and incremental gain answer different questions.}']
    for depth in depths:
        group=[r for r in data['runs'] if r['depth']==depth and r['method']=='revised_bernstein_shape']
        mean=revision_interval(group,lambda r:100*(r['initial_objective']-r['objective']))
        changed=sum(not np.allclose(r['theta'],r['initial'],atol=1e-12,rtol=0) for r in group)
        lines.append(f'At depth {depth}, revised shaped Bernstein has paired refinement gain '
                     f'{fmt_ci(mean)} percentage points and changes {changed}/{len(group)} outputs. ')
        original=[r for r in rows if r['depth']==depth and r['method']=='bernstein_shape']
        original_gain=revision_interval(original,lambda r:100*(r['initial_objective']-r['objective']))
        effect=paired_revision_interval([r for r in rows if r['depth']==depth],'revised_bernstein_shape','bernstein_shape')
        cost=paired_revision_interval([r for r in rows if r['depth']==depth],'revised_bernstein_shape','bernstein_shape',lambda r:r['shots'])
        graph_changes=len({r['graph'] for r in group if not np.allclose(r['theta'],r['initial'],atol=1e-12,rtol=0)})
        old_graph_changes=len({r['graph'] for r in original if not np.allclose(r['theta'],r['initial'],atol=1e-12,rtol=0)})
        lines.append(f'The original shaped Bernstein gain is {original_gain[0]:.6f} percentage points. '
                     f'Using unrounded paired graph records, the revision minus original gain difference is '
                     f'{effect[0]:.6f} (95\\% interval {effect[1]:.6f} to {effect[2]:.6f}) percentage points, '
                     f'and its mean shot difference is {cost[0]:.2f} (95\\% interval {cost[1]:.2f} to {cost[2]:.2f}). '
                     f'The number of unique targets changed is {graph_changes} for the revision and {old_graph_changes} for the original. '
                     'These depth-specific paired comparisons separate gain from avoided expenditure. ')
        for comparator in ['spsa','cobyla']:
            effect=paired_revision_interval([r for r in rows if r['depth']==depth],
                                            'revised_bernstein_shape',comparator)
            lines.append(f'Its paired gain difference from {REVISION_LABELS[comparator]} is '
                         f'{fmt_ci(effect)} percentage points; positive values favor shaped Bernstein. ')
    lines.append('Figure~\\ref{fig:refinement} reports quality, own-anchor gain, actual shots and '
                 'changed-output counts together. Figure~\\ref{fig:efficiency} uses completed incumbent '
                 'events rather than connecting separate budget-cap means: initialization is available '
                 'at zero shots, and an unfinished atomic batch leaves the preceding output available. '
                 'Target curves record first-ever attainment, retain failed runs in their denominator and count initial successes at zero cost. '
                 'The paired shaped-policy difference uses pointwise graph-block bootstrap bands. Flat tails after termination mean that the output remains available without further expenditure. '
                 'Changing a parameter vector does not guarantee objective improvement for an unconstrained baseline. '
                 'The reference objective is best found, not a certified depth-matched optimum; signed deficits remain signed.')
    lines.extend([r'\begin{table}[t]',
                  r'\caption{Matched revision and historical comparators. Values are graph-mean refinement gains and signed reference deficits in percentage points, followed by actual mean optimization shots in thousands. All use the same descriptor anchor per graph. Add 8,192 independent validation shots per output. H and EB denote Hoeffding and empirical Bernstein; original EB is our own comparator. Graph-paired, fixed-bank-stratified 95\% bootstrap intervals appear in the figures and text.}\label{tab:refinement}',
                  r'\centering\small\setlength{\tabcolsep}{4pt}',
                  r'\begin{tabular}{lrrrrrr}\toprule',
                  r' & \multicolumn{2}{c}{Gain} & \multicolumn{2}{c}{Deficit} & \multicolumn{2}{c}{Shots ($10^3$)} \\',
                  r'Policy & $p=2$ & $p=3$ & $p=2$ & $p=3$ & $p=2$ & $p=3$ \\ \midrule'])
    names=[('initialization_only','Initialization only'),('hoeffding_iso','Original H, isotropic'),
           ('bernstein_iso','Original EB, isotropic'),('bernstein_shape','Original EB, shaped'),('revised_hoeffding_iso','Revised H, isotropic'),
           ('revised_hoeffding_shape','Revised H, shaped'),('revised_bernstein_iso','Revised EB, isotropic'),
           ('revised_bernstein_shape','Revised EB, shaped'),('spsa','SPSA'),('cobyla','COBYLA')]
    for method,label in names:
        values=[]
        for metric in [lambda r:100*(r['initial_objective']-r['objective']),
                       lambda r:100*r['signed_reference_gap'],lambda r:r['shots']/1000]:
            for depth in depths:
                group=[r for r in rows if r['depth']==depth and r['method']==method]
                values.append(f'{revision_interval(group,metric)[0]:.3f}')
        lines.append(label+' & '+' & '.join(values)+r' \\')
    lines.extend([r'\bottomrule\end{tabular}\end{table}',
                  r'\paragraph{Stopping status distinguishes limited resolution from adverse evidence.}'])
    from collections import Counter
    statuses=Counter(r['stop'] for r in data['runs'])
    reasons=Counter(r['stop_reason'] for r in data['runs'])
    lines.append('Across the new factorial, terminal statuses are '+
                 ', '.join(f'\\texttt{{{status.replace("_", r"\_")}}}: {count}' for status,count in sorted(statuses.items()))+
                 '. An unresolved comparison does not certify a bad direction. A resolution-limited stop '
                 'reports that the declared model or decision policy cannot resolve another update within its remaining budget or precision limits; '
                 'it does not establish stationarity or optimality.')
    lines.append('The recorded reasons are '+', '.join(f'\\texttt{{{reason.replace("_", r"\_")}}}: {count}' for reason,count in sorted(reasons.items()))+'.')
    decisions=[decision for row in data['runs'] for decision in row['decisions']]
    lines.append(f'Of the {len(decisions)} actual online proposals, '
                 f'{sum(d["true_decrease"]<=0 for d in decisions)} worsen or leave unchanged the exact objective, '
                 f'{sum(d["true_decrease"]>0 and d["acceptance_margin"]<=0 for d in decisions)} improve without a positive sufficient-decrease margin, '
                 f'{sum(d["acceptance_margin"]>0 and d["decision"]=="unresolved" for d in decisions)} have positive margin but remain unresolved, and '
                 f'{sum(d["acceptance_margin"]>=0 and d["decision"]=="accepted" for d in decisions)} are valid accepted proposals. '
                 'These true-margin classifications are retrospective and never supplied to the optimizer.')
    diagnostics=data['model_diagnostics']
    exact=[r for r in diagnostics if r['model_shots']==1024 and r['replicate']==0]
    sampled=[r for r in diagnostics if r['model_shots']==1024]
    lines.extend([r'\paragraph{Exact-data proposals isolate the affine model from sampling error.}',
                  f'The offline diagnostic uses {len(diagnostics)} sampled fits across '
                  f'{len({r["graph"] for r in diagnostics})} fixed target graphs, both depths, both geometries, '
                  'five radii and four shot counts. The design is held fixed within each target, depth and geometry. '
                  f'Of the {len(exact)} distinct exact-data fits, {sum(r["exact_true_decrease"]>0 for r in exact)} '
                  f'improve the objective and {sum(r["exact_acceptance_margin"]>0 for r in exact)} '
                  'have a positive sufficient-decrease margin. '
                  f'At 1,024 shots per point, {sum(r["true_decrease"]>0 for r in sampled)}/{len(sampled)} sampled '
                  f'proposals improve and {sum(r["acceptance_margin"]>0 for r in sampled)}/{len(sampled)} have a positive sufficient-decrease margin. '
                  'These dependent proposal counts describe the chosen diagnostic grid, not independent graph observations. '
                  'Figure~\\ref{fig:components} separates sampling error around the exact affine fit from total '
                  'error against the central-finite-difference reference gradient (step $10^{-5}$). '
                  'The reference derivative is evaluated only offline. Neither lower fitting error nor better '
                  'conditioning alone establishes useful refinement.'])
    diagnostic_shots=sum(batch['shots'] for row in diagnostics for batch in row['model_batches'])
    power_generated_shots=sum(batch['shots'] for row in data['power_samples'] for look in row['observations'] for batch in look['batches'])
    lines.append(f'Generating the model diagnostics uses {diagnostic_shots:,} simulated observations. '
                 f'The frozen-endpoint replay generates {power_generated_shots:,} simulated observations, '
                 'sharing each stored stream across confidence rules and caps. These ideal-sampling offline '
                 'diagnostic costs are separate from online optimizer cost and do not constitute measurement savings.')
    for radius in [min(r['radius'] for r in exact),max(r['radius'] for r in exact)]:
        group=[r for r in exact if r['radius']==radius]
        lines.append(f'At radius {radius:g}, {sum(r["exact_acceptance_margin"]>0 for r in group)}/{len(group)} '
                     'exact-data proposals have positive sufficient-decrease margin. ')
    power=data['power']
    cap=max(r['cap_per_endpoint'] for r in power)
    records=[r for r in power if r['cap_per_endpoint']==cap and r['rule']=='hoeffding']
    lines.extend([r'\paragraph{Frozen endpoint replays measure the cost of decision resolution.}',
                  f'Before replay, {len(records)} actual sampled-model proposals are fixed by target, depth, '
                  'geometry and radius at 1,024 model shots per point, using the first diagnostic replicate. '
                  f'Each condition is replayed {records[0]["repeats"]} times with independent endpoint streams '
                  'and common error allocation across confidence rules. '
                  f'The preceding extended replay uses a {2*cap:,}-shot endpoint-pair cap and five prescribed looks. '
                  'Figure~\\ref{fig:acceptance-power} now compares all budget prefixes under both four-look and five-look allocations. '
                  'Acceptance-stage cost includes unresolved runs at their caps. The analytical Hoeffding proposition remains '
                  'an upper bound for its decision rule, not a universal QAOA shot lower bound.'])
    for rule in ['hoeffding','bernstein']:
        group=[r for r in power if r['cap_per_endpoint']==cap and r['rule']==rule]
        positive=[r for r in group if r['acceptance_margin']>0]
        lines.append(f'For {rule.capitalize()}, the mean acceptance probability over all fixed proposals is '
                     f'{np.mean([r["acceptance_probability"] for r in group]):.3f}, and the mean unresolved '
                     f'fraction is {np.mean([r["unresolved_fraction"] for r in group]):.3f}. ')
        if positive:
            lines.append(f'Among the {len(positive)} positive-margin proposals, the mean acceptance probability '
                         f'is {np.mean([r["acceptance_probability"] for r in positive]):.3f}, with mean '
                         f'endpoint cost {np.mean([r["mean_acceptance_shots"] for r in positive]):,.0f} shots. ')
    lines.extend([r'\paragraph{Paired regime comparisons delimit the evidence.}',
                  'Figure~\\ref{fig:generality} keeps negative effects visible when comparing shaped Bernstein '
                  'with both conventional optimizers by vertex count and with SPSA by depth within graph families. '
                  'The two reference banks belong to different target panels, so these comparisons do not '
                  'identify a bank effect. All intervals average shot repetitions within graph and resample '
                  'graphs within the two fixed banks (10,000 resamples, seed 5713); they are descriptive and '
                  'unadjusted. The experiment supplies no fresh holdout validation of a revised policy, '
                  'large-graph scaling result, structural-shift test, or hardware-noise validation.'])
    stop_start=next(i for i,line in enumerate(lines) if line.startswith(r'\paragraph{Stopping status'))
    old_diagnostic_start=next(i for i,line in enumerate(lines) if line.startswith(r'\paragraph{Exact-data proposals'))
    resolution=[line for line in lines[stop_start:old_diagnostic_start] if 'actual online proposals' not in line]
    historical=lines[old_diagnostic_start:]
    immediate=lines[:stop_start]
    sections={'ResolutionResults':resolution,'HistoricalDiagnostics':historical}
    mechanism_path=CODE_ROOT/'results'/'mechanism.json.gz'
    if mechanism_path.is_file():
        sections.update(mechanism_results(_mechanism_for_report(),data))
    if (CODE_ROOT/'results'/'decision.json.gz').is_file():
        sections['JointDecisionResults']=decision_results(_decision_for_report())
    for macro,parts in sections.items():
        immediate.append('\\newcommand{\\'+macro+'}{%\n'+'\n'.join(parts)+'\n}')
    generated='\n'.join(immediate)
    (REPO_ROOT/'submission'/'tables'/'refinement.tex').write_text(
        '\n'.join(line.rstrip() for line in generated.splitlines())+'\n')


def mechanism_results(data,revision):
    """Raw-derived attribution, predictive checks and resource accounting."""
    ledger=data['ledger_attribution']
    changed=sum(r['noninitial_radius_models']>0 for r in ledger)
    high=sum(r['above_base_shot_models']>0 for r in ledger)
    models=sum(len(r['model_allocations']) for r in ledger)
    lines=[r'\paragraph{Unresolved stopping often precedes increased model allocation.}',
           f'The matched shape-by-rule ledgers show that {changed}/{len(ledger)} runs fit a model away '
           f'from the initial radius ({sum(r["noninitial_radius_models"] for r in ledger)}/{models} fits), '
           f'and {high}/{len(ledger)} runs allocate more than 256 shots per point '
           f'({sum(r["above_base_shot_models"] for r in ledger)}/{models} fits). '
           'Every revised Hoeffding run fits one initial model and performs one full four-look endpoint '
           'comparison before unresolved stopping. Its costs are therefore exactly $5(256)+2(16384)=34048$ '
           'at depth two and $7(256)+2(16384)=34560$ at depth three. '
           'Those savings demonstrate avoided continuation, not an observed improvement from changing '
           'the model-shot allocation on the Hoeffding trajectories.',
           r'\paragraph{A separate factorial isolates sampling allocation from unresolved stopping.}',
           f'The sampling-by-stopping experiment contains {len(data["attribution_runs"])} shaped empirical-Bernstein runs '
           'on the same 16 targets, crossing fixed/radius-dependent model shots with stop/continue '
           'after unresolved evidence. All four cells share anchors, shape, model family, error '
           'allocation, shot cap, initial settings and paired seeds. Continuation holds the radius '
           'and refits from fresh data; it contracts only after certified rejection. This control '
           'is not a replay of the historical rule that contracted after unresolved evidence. '
           'Figure~\\ref{fig:attribution} separates gain and expenditure for these controlled policies.',
           r'\begin{table}[t]',
           r'\caption{Controlled sampling-by-stopping attribution. Gains are percentage points over the common anchor; costs are actual optimization shots in thousands. Every row uses shaped empirical-Bernstein refinement at a 65,536-shot cap, with 8,192 independent validation shots per output. Two shot repetitions are averaged within each of 16 target graphs.}\label{tab:attribution}',
           r'\centering\small\begin{tabular}{lrrrr}\toprule',
           r' & \multicolumn{2}{c}{Gain (pp)} & \multicolumn{2}{c}{Shots ($10^3$)} \\',
           r'Allocation / unresolved policy & $p=2$ & $p=3$ & $p=2$ & $p=3$ \\ \midrule']
    rows=data['attribution_runs']
    identical,total=identical_attribution_outputs(rows)
    for method,label in [('fixed_continue','Fixed / continue'),('fixed_stop','Fixed / stop'),
                         ('radius_continue','Radius / continue'),('radius_stop','Radius / stop')]:
        values=[]
        for value in [lambda r:100*(r['initial_objective']-r['objective']),lambda r:r['shots']/1000]:
            for depth in [2,3]:
                group=[r for r in rows if r['depth']==depth and r['method']==method]
                values.append(f'{revision_interval(group,value)[0]:.4f}')
        lines.append(label+' & '+' & '.join(values)+r' \\')
    lines.append(r'\bottomrule\end{tabular}\end{table}')
    lines.append(f'All four cells return byte-identical parameter vectors in {identical}/{total} matched graph--depth--seed cases. '
                 'In this controlled sample, neither continuation nor radius-dependent allocation changes returned quality; stopping changes expenditure.')
    for depth in [2,3]:
        group=[r for r in rows if r['depth']==depth]
        for allocation in ['fixed','radius']:
            cost=paired_revision_interval(group,allocation+'_stop',allocation+'_continue',lambda r:r['shots'])
            saving=paired_cost_saving_interval(group,allocation+'_stop',allocation+'_continue')
            lines.append(f'At depth {depth} under {allocation} sampling, stopping minus continuation has paired shot difference '
                         f'{cost[0]:.2f} (95\\% interval {cost[1]:.2f} to {cost[2]:.2f}), '
                         f'a {saving[0]:.2f}\\% reduction in mean optimization shots '
                         f'(paired graph-bootstrap interval {saving[1]:.2f}\\% to {saving[2]:.2f}\\%). ')
        for policy in ['stop','continue']:
            effect=paired_revision_interval(group,'radius_'+policy,'fixed_'+policy)
            cost=paired_revision_interval(group,'radius_'+policy,'fixed_'+policy,lambda r:r['shots'])
            lines.append(f'At depth {depth} with unresolved {policy}, radius-dependent minus fixed '
                         f'allocation has gain difference {fmt_ci(effect)} percentage points '
                         f'and mean shot difference {cost[0]:.2f}. ')
    lines.append('Percentage savings divide the difference of graph-mean costs by the continuation mean; '
                 'both costs are resampled together within each fixed bank. These reductions concern '
                 'the specified hold-radius continuation policy at the shared cap. After a first complete '
                 'test and two initial-radius models, only 30,208 shots at depth two or 29,184 at depth three '
                 'remain for the next endpoint pair, below its 32,768-shot final look. '
                 'The comparison does not establish an optimal use of that remaining budget.')
    lines.extend([r'\paragraph{Online power uses its own four-look confidence allocation.}',
                  'Figure~\\ref{fig:acceptance-power} holds fixed the 58 positive-margin proposals '
                  'and the 102 nonpositive-margin proposals, with equal proposal weights at every budget. '
                  'The online row uses four prescribed looks up to 16,384 shots per endpoint; the extended '
                  'row uses five looks up to 65,536. The online acceptance panel is magnified to 0--0.06; '
                  'the extended acceptance panel spans 0--1. Budget positions are numerical coordinates '
                  'on a base-two logarithmic axis. Both schedules are recomputed from the same stored independent '
                  'streams, so this comparison needs no additional sampled observations. Shading gives '
                  'conservative pointwise 95\\% Hoeffding bounds for each fixed-cohort replay mean, allowing '
                  'different endpoint distributions across proposals. Those intervals quantify Monte Carlo '
                  'replay uncertainty conditional on the fixed proposals, not uncertainty over target graphs.'])
    for key,label in [('online_power','online four-look'),('extended_power','extended five-look')]:
        cap=max(r['endpoint_pair_budget'] for r in data[key])
        for rule in ['hoeffding','bernstein']:
            group=[r for r in data[key] if r['rule']==rule and r['endpoint_pair_budget']==cap and r['acceptance_margin']>0]
            mean,lo,hi=replay_mean_interval(group,'acceptance_probability',cap)
            lines.append(f'At the {cap:,}-shot pair cap, {label} {rule.capitalize()} has mean positive-margin '
                         f'acceptance frequency {mean:.4f} (pointwise interval {lo:.4f} to {hi:.4f}), '
                         f'unresolved fraction {np.mean([r["unresolved_fraction"] for r in group]):.4f}, '
                         f'and mean actual endpoint expenditure {np.mean([r["mean_acceptance_shots"] for r in group]):,.2f} shots. ')
    n=data['online_power'][0]['repeats']
    upper=1-.05**(1/n)
    lines.append(f'For one pair, zero acceptances in {n} independent replays has exact one-sided '
                 f'95\\% binomial upper limit {upper:.5f} ({100*upper:.3f}\\%). Zero observed frequency '
                 'therefore does not establish zero probability or empirically validate an analytical '
                 'probability of order $10^{-6}$.')
    diagnostics=data['model_diagnostics']
    old=[r for r in data['instances'] if r['split']=='diagnostic']
    fresh=[r for r in data['instances'] if r['split']=='fresh']
    max_error=max(r['difference_norm'] for r in data['gradient_checks'])
    mantissa,exponent=f'{max_error:.2e}'.split('e')
    lines.extend([r'\paragraph{The implemented schedule is tested against its variance prediction.}',
                  f'The mechanism study uses {len(old)} original diagnostic graphs from panel zero and '
                  f'{len(fresh)} newly generated targets nonisomorphic to all previous bank, validation '
                  'and test graphs. Figure~\\ref{fig:components} displays shaped models on the original '
                  'eight, with depths two and three equally averaged within each graph before bootstrap '
                  'resampling. Full graph identities and their split membership are recorded in the saved protocol.',
                  f'Central-difference gradients at steps $10^{{-5}}$ and $5\\times10^{{-6}}$ differ by '
                  f'at most ${mantissa}\\times10^{{{int(exponent)}}}$ in Euclidean norm across all targets and depths. '
                  'This check supports the numerical reference used in the bias comparison; it does not '
                  'turn a numerical derivative into an analytic exact-gradient formula.',
                  f'The {len(diagnostics)} target/depth/shape/radius/shot-count conditions each use '
                  f'{data["config"]["fit_repetitions"]} independent sampled fits. Root-mean-square error '
                  'means the square root after averaging squared Euclidean errors, including graph '
                  'averaging; it is not the arithmetic mean of error norms. Exact-data bias floors and '
                  'the propagated covariance predict total mean-square error. '
                  'The actual schedule samples 4096, 1024, 256, 64 and 16 shots per point at radii '
                  '0.025, 0.05, 0.1, 0.2 and 0.4. Its cost is multiplied by $2p+1$ fitting points. '
                  'Graph-level bands are pointwise 95\\% intervals conditional on the fixed bank and '
                  'sampled designs; they do not provide simultaneous radius-wise coverage.'])
    fresh_rows=[r for r in diagnostics if r['split']=='fresh']
    ratio=np.mean([r['observed_sampling_mse'] for r in fresh_rows])/np.mean([r['predicted_sampling_mse'] for r in fresh_rows])
    total_ratio=np.mean([r['observed_total_mse'] for r in fresh_rows])/np.mean([r['predicted_total_mse'] for r in fresh_rows])
    scheduled=[r for r in fresh_rows if r['is_schedule']]
    schedule_sampling=np.mean([r['observed_sampling_mse'] for r in scheduled])/np.mean([r['predicted_sampling_mse'] for r in scheduled])
    schedule_total=np.mean([r['observed_total_mse'] for r in scheduled])/np.mean([r['predicted_total_mse'] for r in scheduled])
    lines.append(f'Across the fresh-graph model grid, the ratios of mean observed to mean predicted '
                 f'sampling and total MSE are {ratio:.6f} and {total_ratio:.6f}, respectively. '
                 f'On the implemented schedule alone, the corresponding ratios are {schedule_sampling:.6f} and {schedule_total:.6f}. '
                 'Both summaries equally average the fresh targets, depths and shapes. Agreement checks '
                 'the standard conditional second-moment identity using exact simulator means, variances '
                 'and numerical reference derivatives; it is not a new variance formula or a measurement-only prediction. '
                 'Gaussian positive-margin prediction additionally '
                 'approximates the distribution of fitted objective means and the nonlinear proposal map.')
    lookup={r['id']:r for r in data['predictions']}
    matched=[]
    for row in data['schedule_proposals']:
        if row['split']=='fresh' and row['shape_kind']=='shape':
            prediction=lookup[row['prediction_id']]['gaussian_positive_margin_probability']
            matched.append({'graph':row['graph'],'error':row['observed_positive_margin_fraction']-prediction})
    mean,lo,hi=graph_diagnostic_interval(matched,lambda r:100*abs(r['error']))
    lines.extend([r'\paragraph{Fresh-graph prediction is simulator-informed and conditional.}',
                  'All fitting maps, covariance predictions and Gaussian proposal predictions were '
                  'frozen before independent multinomial observations. The fresh targets use the same '
                  'fixed panel-zero donor bank and no fresh-target tuning. '
                  f'The fresh-graph study compares {data["config"]["gaussian_repetitions"]} Gaussian '
                  f'proposal draws per condition with {data["config"]["proposal_repetitions"]} independent '
                  'sampled proposals along the implemented radius schedule. '
                  f'The mean absolute difference in positive-margin probability across shaped fresh '
                  f'conditions is {mean:.3f} percentage points (graph-bootstrap interval {lo:.3f} to {hi:.3f}). '
                  'This discrepancy includes Gaussian approximation error and finite replay noise. '
                  'The predictor uses exact simulator objective means, variances and trial expectations. '
                  'It is an oracle-informed mechanism test, not a measurement-implementable regime '
                  'selector, fresh-graph optimizer benchmark, or hardware validation.'])
    residuals=fresh_prediction_residuals(data)
    references=[]
    for depth in [2,3]:
        for radius in sorted({r['radius'] for r in residuals}):
            group=[r for r in residuals if r['depth']==depth and r['radius']==radius]
            lower,upper=conditional_mc_reference(group,data['config']['proposal_repetitions'],
                                                  data['config']['gaussian_repetitions'])
            observed=float(np.mean([r['error'] for r in group]))
            references.append(lower-1e-12<=observed<=upper+1e-12)
    lines.append('Figure~\\ref{fig:attribution} separates signed observed-minus-Gaussian residuals '
                 'from the absolute-error summary. Error bars resample target graphs. Gray regions are '
                 'a different, exact conditional Monte Carlo reference: under equal Gaussian and sampled '
                 'success probabilities within a graph, its 32-sample success count is hypergeometric '
                 'conditional on the combined successes in 32+256 draws. Convolving the eight graph '
                 'distributions gives a central 95\\% reference for their equal-weight signed difference. '
                 f'{sum(references)}/{len(references)} plotted depth/radius differences fall in this reference. '
                 'This pointwise null range is neither a confidence interval for systematic approximation '
                 'error nor uncertainty over a graph population. A zero-success condition gives a '
                 'degenerate conditional reference, not evidence of zero underlying probability.')
    costs=data['accounting']
    lines.append(f'The additional attribution consumes {costs["attribution_optimization_shots"]:,} '
                 f'optimization and {costs["attribution_validation_shots"]:,} independent validation shots. '
                 f'The mechanism observations require {costs["mechanism_observation_shots"]:,} simulated '
                 f'finite-shot outcomes; the Gaussian diagnostic uses {costs["gaussian_samples"]:,} draws '
                 f'and prediction/reference construction uses {costs["prediction_exact_calls"]:,} exact '
                 f'simulator queries, plus {costs["scheduled_proposal_diagnostic_exact_calls"]:,} exact sampled-proposal '
                 'diagnostic objective evaluations. Those offline diagnostic costs are not credited as optimizer savings.')
    starts={name:next(i for i,line in enumerate(lines) if line.startswith(prefix)) for name,prefix in {
        'attribution':r'\paragraph{A separate factorial',
        'power':r'\paragraph{Online power',
        'schedule':r'\paragraph{The implemented schedule',
        'prediction':r'\paragraph{Fresh-graph prediction'}.items()}
    return {'RadiusResults':lines[:starts['attribution']]+lines[starts['schedule']:starts['prediction']],
            'PowerResults':lines[starts['power']:starts['schedule']],
            'MechanismResults':lines[starts['attribution']:starts['power']]+lines[starts['prediction']:]}


def decision_results(data):
    """Summarize matched-schedule utility and post-hoc predictive controls."""
    calibration=[r for r in data['calibration_records'] if r['shape_kind']=='shape']
    lines=[r'\paragraph{Simple prediction controls delimit the oracle-informed comparison.}',
           'The post-hoc radius-only and radius-plus-depth controls estimate condition means from '
           'the eight original diagnostic graphs, then evaluate the same shaped conditions on the eight '
           'fresh targets. Neither uses fresh outcomes to fit its condition means. They test whether '
           'exact-target-informed Gaussian predictions describe variation beyond these simpler controls; '
           'the comparison does not supply a measurement-only prediction rule. ']
    for predictor,label in [('gaussian','Gaussian'),('radius_depth','radius plus depth'),('radius_only','radius only')]:
        group=[r for r in calibration if r['predictor']==predictor]
        interval=graph_diagnostic_interval(group,lambda r:100*r['abs_error'])
        brier=graph_diagnostic_interval(group,lambda r:r['brier_score'])[0]
        lines.append(f'The {label} predictor has mean absolute frequency difference {fmt_ci(interval)} '
                     f'percentage points and mean Bernoulli Brier score {brier:.6f}. ')
    null=next(r for r in data['calibration_null'] if r['shape_kind']=='shape')
    lines.append('Under the within-condition equal-probability null, the exact conditional expectation '
                 f'of the absolute frequency difference is {100*null["expected_null_mae"]:.4f} percentage points; '
                 f'{data["config"]["null_repetitions"]:,} independent conditional null draws give a central '
                 f'95\\% reference of {100*null["null_mae_quantiles"][0]:.4f}--'
                 f'{100*null["null_mae_quantiles"][-1]:.4f} percentage points for its mean. '
                 f'The observed {100*null["observed_mae"]:.4f} lies inside this Monte Carlo reference. '
                 'This is compatible with finite simulation noise at the available resolution; it does '
                 'not prove equality or identify pure Gaussian approximation error.')
    rows=data['joint_records']
    cap=max(r['endpoint_pair_budget'] for r in rows)
    lines.extend([r'\paragraph{Matched proposal and decision measurements reveal a conditional operating window.}',
                  'Figure~\\ref{fig:joint-utility} applies the four-look endpoint rules to the same fresh '
                  'shaped radius/depth conditions used for proposal prediction, using the actual '
                  '4096, 1024, 256, 64 and 16 model shots per point. Each condition retains all 256 '
                  'Gaussian proposals with one independent endpoint replay each, and all 32 sampled '
                  'proposals with eight independent endpoint replays each. The eight replays do not '
                  'create eight new model fits. All margin signs and unsuccessful decisions remain in '
                  'the denominator. Accepted true gain is zero on a failed decision and retains its '
                  'sign on acceptance; cost charges one model plus the actual endpoint measurements '
                  'for each attempt. Graph-level intervals are pointwise and descriptive over the eight '
                  'fixed-bank fresh targets, not independent-observation intervals over endpoint replays.'])
    for depth in [2,3]:
        for population,label in [('sampled','sampled observations'),('gaussian','Gaussian forecasts')]:
            group=[r for r in rows if r['depth']==depth and r['radius']==.1 and r['population']==population
                   and r['rule']=='bernstein' and r['endpoint_pair_budget']==cap]
            gain=graph_diagnostic_interval(group,lambda r:r['mean_accepted_true_gain_pp'])
            cost=graph_diagnostic_interval(group,lambda r:r['mean_total_shots'])
            probability=graph_diagnostic_interval(group,lambda r:100*r['acceptance_probability'])
            positive=sum(r['mean_accepted_true_gain_pp']>0 for r in group)
            lines.append(f'At depth {depth}, radius 0.1 and the {cap:,}-shot pair cap, {label} give '
                         f'Bernstein accepted true gain {gain[0]:.6f} percentage points '
                         f'(95\\% graph interval {gain[1]:.6f} to {gain[2]:.6f}), '
                         f'acceptance {probability[0]:.4f}\\%, and mean total cost {cost[0]:,.2f} shots '
                         f'(95\\% graph interval {cost[1]:,.2f} to {cost[2]:,.2f}). '
                         f'Positive accepted gain occurs on {positive}/{len(group)} target graphs. ')
    smallest=min(r['radius'] for r in rows)
    small=[r for r in rows if r['radius']==smallest and r['population']=='sampled' and
           r['rule']=='bernstein' and r['endpoint_pair_budget']==cap]
    for depth in [2,3]:
        group=[r for r in small if r['depth']==depth]
        gain=graph_diagnostic_interval(group,lambda r:r['mean_accepted_true_gain_pp'])[0]
        cost=graph_diagnostic_interval(group,lambda r:r['mean_total_shots'])[0]
        lines.append(f'At radius {smallest:g}, sampled Bernstein has accepted gain {gain:.6f} '
                     f'percentage points at depth {depth}, despite mean total cost {cost:,.0f} shots. ')
    hoeffding=[r for r in rows if r['rule']=='hoeffding']
    if all(r['acceptance_probability']==0 for r in hoeffding):
        lines.append('Hoeffding accepts no proposal in either simulated population at any tested allowance. ')
    lines.append('High positive-margin frequency alone therefore does not identify the largest '
                 'certified gain: proposal magnitude, decision resolution and model expenditure jointly '
                 'matter. This post-hoc one-attempt diagnostic remains conditioned on exact target '
                 'information and a fixed bank. Its gain is not a held-out complete-optimizer performance '
                 'result or a comparison with SPSA or COBYLA.')
    costs=data['accounting']
    lines.append(f'The joint diagnostic generates {costs["new_endpoint_simulated_shots"]:,} new simulated '
                 f'endpoint outcomes across {costs["endpoint_replays"]:,} replays and '
                 f'{costs["exact_distribution_queries"]:,} exact distribution queries. '
                 'Its model proposals are reused from the saved mechanism experiment; no new model '
                 'measurements are generated. Per-attempt cost nevertheless charges the saved model '
                 'allocation once. Rules and budget prefixes reuse each new endpoint stream, so their '
                 'reported costs must not be summed as independent offline sampling expense.')
    return lines


def figure_manifest(data, data_path, target):
    descriptions = {
        'quality.pdf': {'quantity': 'graph-mean signed reference deficit and graph bootstrap interval', 'depths': [1,2], 'budget': max(data['config']['budgets'])},
        'transfer.pdf': {'quantity': 'initial signed reference deficit by prior', 'depths': [1,2]},
        'accounting.pdf': {'quantity': 'stacked model, endpoint, baseline objective and final evaluation shots, including sampling-by-stopping controls', 'depths': [1,2,3], 'budgets': data['config']['budgets']},
        'conditioning.pdf': {'quantity': 'historical trial-weighted design-condition and verified physical-fitting-map-norm ECDFs, plus certificate saturation bars', 'depths': [1,2]}}
    if (target/'transport.pdf').is_file():
        descriptions['transport.pdf']={'quantity':'solver feasibility, donor changes, and quality against descriptor/original references on matched valid target sets'}
    if (CODE_ROOT/'results'/'refinement.json.gz').is_file():
        descriptions.update({name:{'quantity':description,'study':'post-hoc matched refinement revision'}
            for name,description in {
                'refinement.pdf':'initial and final deficit, paired own-anchor gain, actual shots and changed output counts',
                'efficiency.pdf':'actual incumbent gain, first-ever target attainment and paired shaped-policy curves with graph-block bands',
                'acceptance_power.pdf':'fixed-cohort power, unresolved fraction and actual expenditure across endpoint budgets, online four-look and extended five-look allocations',
                'components.pdf':'graph-level RMS bias-variance, positive margin and magnitude, actual radius schedule cost and prediction',
                'generality.pdf':'supporting paired refinement-gain effects versus both conventional comparators by graph size and depth',
                'attribution.pdf':'controlled sampling-by-stopping gains/costs, paired cost effects, and signed fresh prediction residuals with graph and conditional Monte Carlo uncertainty'}.items()})
    if (target/'decision.pdf').is_file():
        descriptions['decision.pdf']={'quantity':'fresh scheduled positive-margin probability, accepted true gain including failures, and model-plus-endpoint cost; oracle-informed forecasts versus independent observations',
                                      'study':'post-hoc matched fresh-schedule decision experiment'}
    files = [{'path': (target/name).relative_to(REPO_ROOT).as_posix(), 'sha256': digest(target/name), **description}
             for name,description in descriptions.items()]
    sources = {name: digest(CODE_ROOT/name) for name in
               ['src/gcqaoa/report.py','src/gcqaoa/artifacts.py','src/gcqaoa/provenance.py']}
    manifest = {'provenance': lineage(data,data_path), 'analysis_source_sha256': sources,
                'numerical_source_sha256': data['source_sha256'],
                'selection': {'graph_ids': sorted(row['id'] for row in data['instances'] if row['split']=='test'),
                              'depths': data['config']['depths'], 'budgets': data['config']['budgets'],
                              'methods': data['config']['methods'], 'repetitions': data['config']['repetitions']},
                'bootstrap': {'seed':5713,'resamples':10000,'unit':'graph mean over shot repetitions',
                              'interval':'95 percent percentile; unadjusted descriptive comparisons'},
                'figures':files}
    if Path(data_path).resolve()==CANONICAL_DATA:
        table=REPO_ROOT/'submission'/'tables'/'results.tex'
        manifest['tables']=[{'path':table.relative_to(REPO_ROOT).as_posix(),'sha256':digest(table),
                            'quantity':'generated study result paragraphs and complete depth-two ablation table',
                            'depth':2,'budget':max(data['config']['budgets']),
                            'methods':data['config']['methods']}]
        manifest['website']={'path':'index.html','block_sha256':hashlib.sha256(website_results(data).encode()).hexdigest(),
                             'quantity':'canonical graph bootstrap intervals and measured online shots',
                             'depth':2,'budget':max(data['config']['budgets'])}
        manifest['additional_studies']={name:digest(CODE_ROOT/'results'/name)
                                        for name in ('followup.json.gz','transport.json.gz','refinement.json.gz','mechanism.json.gz','decision.json.gz')
                                        if (CODE_ROOT/'results'/name).is_file()}
        revision_path=CODE_ROOT/'results'/'refinement.json.gz'
        if revision_path.is_file():
            table=REPO_ROOT/'submission'/'tables'/'refinement.tex'
            manifest['tables'].append({'path':table.relative_to(REPO_ROOT).as_posix(),
                                      'sha256':digest(table),'quantity':'generated post-hoc refinement findings and complete factorial table'})
        manifest['followup_bootstrap']={'seed':5713,'resamples':10000,
                                       'unit':'graph mean over shot repetitions',
                                       'strata':'two fixed reference banks',
                                       'interval':'95 percent percentile; unadjusted descriptive comparisons'}
    write_json(Path(data_path).resolve().parent/'aggregate'/'figure_manifest.json',manifest)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=CANONICAL_DATA)
    parser.add_argument('--config',type=Path,
                        help='require these settings to match the selected saved study; does not run experiments')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--analysis-only',action='store_true',help='write CSVs and canonical manuscript results')
    modes.add_argument('--figures-only',action='store_true',help='render figures and their provenance manifest')
    args=parser.parse_args(argv)
    data=load_study(args.data)
    if args.config is not None:
        try:
            requested_config=json.loads(args.config.read_text())
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f'cannot read configuration: {error}')
        if requested_config != data['config']:
            parser.error('--config does not match the saved study selected by --data')
    canonical=args.data.resolve()==CANONICAL_DATA
    if canonical:
        update_website(data)
    if not args.figures_only:
        analyze(data,args.data)
        if canonical:
            write_results(data)
    if not args.analysis_only:
        if canonical and args.figures_only:
            write_results(data)
        target=REPO_ROOT/'submission'/'figures' if canonical else args.data.resolve().parent/'figures'
        render(data,target)
        figure_manifest(data,args.data,target)
    print(f"Generated requested artifacts from {len(data['runs'])} runs ({data['execution_run_id']}).")

if __name__=='__main__':
    main()
