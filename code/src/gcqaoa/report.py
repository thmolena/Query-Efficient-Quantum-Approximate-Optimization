"""Regenerate all manuscript statistics and vector figures from the saved study."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
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
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.5),sharey=True)
    for ax,p in zip(axes,[1,2]):
        for j,(method,color) in enumerate(zip(methods,COLORS)):
            values=graph_means(selected(data,method,p,budget),'signed_reference_gap')
            mean,lo,hi=ci([100*v for v in values.values()])
            jitter=np.linspace(-.18,.18,len(values))
            ax.scatter([100*v for v in values.values()],j+jitter,s=7,color=color,alpha=.25,edgecolors='none')
            ax.errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=2,ms=4)
        ax.axvline(0,color='.7',lw=.7)
        ax.axvline(2,color='.5',lw=.8,ls=':')
        ax.set_title(f'Depth p = {p}')
        ax.set_xlabel('Expected cut deficit (percentage points)')
        ax.set_yticks(range(len(methods)),[LABELS[m] for m in methods])
        ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis()
    fig.suptitle(f'Held-out quality at a cap of {budget:,} optimization shots',fontsize=10)
    fig.tight_layout()
    fig.savefig(target/'quality.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0))
    kinds=['uniform','descriptor','gw','local','shuffled']
    colors=['#526777','#0072B2','#D55E00','#CC79A7','#999999']
    for ax,p in zip(axes,[1,2]):
        for j,(kind,color) in enumerate(zip(kinds,colors)):
            values=[]
            for key,priors in data['priors'].items():
                if key.endswith(f':p{p}'):
                    values.append(100*(priors[kind]['initial_objective']-data['references'][key]['objective']))
            mean,lo,hi=ci(values)
            jitter=np.linspace(-.16,.16,len(values))
            ax.scatter(j+jitter,values,color=color,alpha=.35,s=9,edgecolors='none')
            ax.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',color=color,capsize=3,lw=1,ms=4)
        ax.set_xticks(range(5),['Fixed donor','Descriptor','GW','Local TV','Shuffled'],rotation=25,ha='right')
        ax.set_title(f'Depth p = {p}')
        ax.axhline(2,color='.5',ls=':',lw=.8)
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Initial cut deficit (percentage points)')
    fig.suptitle('Reference transfer before target-graph optimization',fontsize=10)
    fig.tight_layout()
    fig.savefig(target/'transfer.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.1,3.1))
    for method,color in zip(methods,COLORS):
        xs,ys=[],[]
        for budget in data['config']['budgets']:
            rows=selected(data,method,2,budget)
            xs.append(np.mean([r['shots'] for r in rows])/1000)
            ys.append(np.mean([r['expected_ratio'] for r in rows]))
        axes[0].plot(xs,ys,'o-',color=color,label=LABELS[method],ms=3,lw=1)
    axes[0].set_xlabel('Mean optimization shots (thousands)')
    axes[0].set_ylabel('Expected cut / exact MaxCut')
    axes[0].set_title('Depth 2: measured quality and cost')
    methods2=['descriptor_iso','descriptor_shape','gw_shape','no_repair','fixed_radius','fixed_shots']
    for j,method in enumerate(methods2):
        rows=selected(data,method,2,max(data['config']['budgets']))
        stages={stage:np.mean([sum(e['shots'] for e in r['evaluations'] if e['stage']==stage) for r in rows])/1000 for stage in ['model','acceptance']}
        axes[1].barh(j,stages['model'],color='#0072B2',label='Model' if j==0 else None)
        axes[1].barh(j,stages['acceptance'],left=stages['model'],color='#E69F00',label='Acceptance' if j==0 else None)
    axes[1].set_yticks(range(len(methods2)),[LABELS[m] for m in methods2])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean shots by stage (thousands)')
    axes[1].set_title('Cost of controlled acceptance')
    axes[1].legend(frameon=False,loc='lower right')
    axes[0].legend(frameon=False,fontsize=6,loc='best')
    fig.tight_layout()
    fig.savefig(target/'accounting.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    for method,color in [('descriptor_shape','#0072B2'),('no_repair','#D55E00')]:
        rows=selected(data,method,2,max(data['config']['budgets']))
        cond=sorted(d['condition'] for r in rows for d in r['decisions'])
        axes[0].step(cond,np.arange(1,len(cond)+1)/len(cond),where='post',label=LABELS[method],color=color)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Weighted design condition number')
    axes[0].set_ylabel('Empirical cumulative fraction')
    axes[0].legend(frameon=False)
    for p,color in [(1,'#0072B2'),(2,'#D55E00')]:
        values=[x for key,priors in data['priors'].items() if key.endswith(f':p{p}') for x in priors['local']['scores']]
        axes[1].hist(np.clip(values,0,1),bins=np.linspace(0,1,11),alpha=.6,color=color,label=f'Depth {p}')
    axes[1].set_xlabel('Exact edge-neighborhood total variation')
    axes[1].set_ylabel('Target/reference pairs')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(target/'conditioning.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
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


def figure_manifest(data, data_path, target):
    descriptions = {
        'quality.pdf': {'quantity': 'graph-mean signed reference deficit and graph bootstrap interval', 'depths': [1,2], 'budget': max(data['config']['budgets'])},
        'transfer.pdf': {'quantity': 'initial signed reference deficit by prior', 'depths': [1,2]},
        'accounting.pdf': {'quantity': 'expected approximation ratio versus optimization shots and model/acceptance shot allocation', 'depths': [2], 'budgets': data['config']['budgets']},
        'conditioning.pdf': {'quantity': 'design condition-number empirical CDF and exact local-neighborhood total variation histogram', 'depths': [1,2]}}
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
                                        for name in ('followup.json.gz','transport.json.gz')
                                        if (CODE_ROOT/'results'/name).is_file()}
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
