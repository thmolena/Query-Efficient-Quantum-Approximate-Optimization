"""Frozen, leakage-controlled MaxCut QAOA study; run from the repository root."""
from __future__ import annotations

import argparse
import gzip
import json
import platform
from pathlib import Path
import time
import numpy as np
import networkx as nx
import scipy
from scipy.optimize import minimize
from .qaoa import MaxCutQAOA, exact_maxcut, normalized_shortest_path_matrix, local_tv
from .search import descriptor, graph_prior, structural_score, optimize, wrap

from .provenance import PAPER_CONFIG, execution_metadata, run_id, safe_output


def make_graph(family, n, seed):
    for attempt in range(1000):
        state = seed+attempt
        if family == 'erdos_renyi':
            graph = nx.gnp_random_graph(n, .4, seed=state)
        elif family == 'regular':
            graph = nx.random_regular_graph(3, n, seed=state)
        elif family == 'barabasi_albert':
            graph = nx.barabasi_albert_graph(n, 2, seed=state)
        elif family == 'watts_strogatz':
            graph = nx.watts_strogatz_graph(n, 4, .35, seed=state)
        else:
            raise ValueError(family)
        if nx.is_connected(graph):
            return graph
    raise RuntimeError('Failed connected graph generation')


def instances(config):
    rows, graphs = [], []
    rng = np.random.default_rng(config['seed'])
    for split in ['bank', 'test']:
        for family in config['families']:
            for n in config[split+'_sizes']:
                for replicate in range(config[split+'_per_family_size']):
                    for _ in range(1000):
                        seed = int(rng.integers(1, 2**30))
                        graph = make_graph(family, n, seed)
                        # Freeze disjointness up to graph isomorphism, across
                        # all bank/test families. No parameter labels consulted.
                        if not any(len(g)==n and nx.is_isomorphic(graph,g) for g in graphs):
                            break
                    else:
                        raise RuntimeError('Insufficient nonisomorphic instances')
                    graphs.append(graph)
                    rows.append({'id': f'{split}_{family}_{n}_{replicate}',
                                 'split': split, 'family': family, 'n': n,
                                 'seed': seed, 'edges': sorted([sorted(e) for e in graph.edges()])})
    return rows


def graph_from(row):
    graph = nx.Graph()
    graph.add_nodes_from(range(row['n']))
    graph.add_edges_from(row['edges'])
    return graph


def reference(circuit, p, config, seed):
    rng = np.random.default_rng(seed)
    points = []
    start = time.perf_counter()
    for i in range(config['reference_starts']):
        # One established low-depth style schedule and five independent starts.
        if i == 0:
            theta = np.r_[np.linspace(.35,.8,p), np.linspace(.35,.15,p)]
        else:
            theta = np.r_[rng.uniform(-np.pi,np.pi,p), rng.uniform(-np.pi/4,np.pi/4,p)]
        result = minimize(circuit.objective, theta, method='BFGS',
                          options={'maxiter': config['reference_maxiter'], 'gtol':1e-7})
        points.append({'theta': wrap(result.x).tolist(), 'objective': float(result.fun),
                       'calls': int(result.nfev), 'success': bool(result.success),
                       'message': str(result.message)})
    best = min(points, key=lambda row:row['objective'])
    return {'theta': best['theta'], 'objective':best['objective'], 'starts':points,
            'exact_calls':sum(row['calls'] for row in points),
            'seconds':time.perf_counter()-start}


def run(config, config_path=PAPER_CONFIG):
    start = time.perf_counter()
    data = {'config':config, **execution_metadata(config_path),
            'environment':{'python':platform.python_version(), 'numpy':np.__version__,
                           'scipy':scipy.__version__, 'networkx':nx.__version__,
                           'machine':platform.machine(), 'system':platform.system()},
            'instances':instances(config), 'references':{}, 'priors':{}, 'runs':[]}
    for idx,row in enumerate(data['instances']):
        graph = graph_from(row)
        circuit = MaxCutQAOA(graph)
        row['maxcut'] = exact_maxcut(graph)[0]
        row['descriptor'] = descriptor(graph).tolist()
        row['structure'] = normalized_shortest_path_matrix(graph).tolist()
        for p in config['depths']:
            key = f"{row['id']}:p{p}"
            data['references'][key] = reference(circuit,p,config,config['seed']+10000*idx+p)
        print(f"reference {idx+1}/{len(data['instances'])}: {row['id']}",flush=True)
    banks = {p:[dict(row,theta=data['references'][f"{row['id']}:p{p}"]['theta'])
                for row in data['instances'] if row['split']=='bank'] for p in config['depths']}
    for idx,row in enumerate([r for r in data['instances'] if r['split']=='test']):
        graph = graph_from(row)
        circuit = MaxCutQAOA(graph)
        gw_start = time.perf_counter()
        scored = [structural_score(np.array(row['structure']),np.array(b['structure']))
                  for b in banks[config['depths'][0]]]
        scores = [item[0] for item in scored]
        gw_seconds = time.perf_counter()-gw_start
        shuffle = np.random.default_rng(config['seed']+idx+777).permutation(len(scores))
        for p in config['depths']:
            key = f"{row['id']}:p{p}"
            priors = {kind:graph_prior(graph,banks[p],kind,scores=scores)
                      for kind in ['uniform','descriptor','gw']}
            local_start = time.perf_counter()
            local_scores = [local_tv(graph,graph_from(b),p) for b in banks[p]]
            local_seconds = time.perf_counter()-local_start
            priors['local'] = graph_prior(graph,banks[p],'local',scores=local_scores)
            priors['shuffled'] = graph_prior(graph,banks[p],'gw',scores=scores,shuffle=shuffle)
            data['priors'][key] = {kind:dict(info,theta=x.tolist(),initial_objective=circuit.objective(x))
                                  for kind,(x,B,info) in priors.items()}
            data['priors'][key]['gw']['solver'] = [item[1] for item in scored]
            data['priors'][key]['gw']['structural_seconds'] = gw_seconds
            data['priors'][key]['local']['structural_seconds'] = local_seconds
            ref = data['references'][key]['objective']
            for rep in range(config['repetitions']):
                seed = config['seed']+100000*idx+1000*p+rep
                for budget in config['budgets']:
                    for method in config['methods']:
                        kind = 'gw' if method.startswith('gw') else 'descriptor'
                        if method in ['uniform_iso','random_iso']:
                            kind='uniform'
                        elif method=='shuffled_gw':
                            kind='shuffled'
                        elif method=='local_iso':
                            kind='local'
                        x,B,info = priors[kind]
                        x,B = x.copy(),B.copy()
                        if method in ['uniform_iso','random_iso','descriptor_iso','gw_iso','local_iso','cobyla','spsa']:
                            B=np.eye(2*p)
                        if method=='random_iso':
                            init_rng=np.random.default_rng(seed+88)
                            x=np.r_[init_rng.uniform(-np.pi,np.pi,p),init_rng.uniform(-np.pi/4,np.pi/4,p)]
                        elif method=='bad_prior':
                            x=wrap(x+np.r_[np.full(p,1.2),np.full(p,.35)])
                        result=optimize(circuit,x,B,config['optimizer'],method,seed,budget)
                        # These exact diagnostics are inaccessible to the optimizer.
                        result['initial_objective']=circuit.objective(x)
                        result['objective']=circuit.objective(result['theta'])
                        result['reference_objective']=ref
                        result['expected_ratio']=-result['objective']*circuit.m/row['maxcut']
                        result['signed_reference_gap']=result['objective']-ref
                        for inc in result['incumbents']:
                            inc['objective']=circuit.objective(inc['theta'])
                        hitting=[inc['shots'] for inc in result['incumbents'] if inc['objective']<=ref+config['target_gap']]
                        result['hitting_shots']=min(hitting) if hitting else None
                        validation_rng=np.random.default_rng(seed+budget+9000000)
                        mean,var=circuit.sample(result['theta'],config['validation_shots'],validation_rng)
                        result['validation']={'shots':config['validation_shots'],'calls':1,'jobs':1,
                                              'mean':float(mean),'variance_of_mean':float(var)}
                        result.update({'graph':row['id'],'depth':p,'replicate':rep,
                                       'method':method,'budget':budget,'seed':seed})
                        result['run_id'] = run_id(data['execution_run_id'], result)
                        data['runs'].append(result)
            print(f"experiment {idx+1}/24 p={p}: {len(data['runs'])} runs",flush=True)
    data['seconds']=time.perf_counter()-start
    return data


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PAPER_CONFIG)
    parser.add_argument('--output',type=Path,help='output under code/results; smoke and pilot cannot overwrite the paper bundle')
    args=parser.parse_args(argv)
    try:
        target=safe_output(args.config,args.output)
    except ValueError as error:
        parser.error(str(error))
    config=json.loads(args.config.read_text())
    data=run(config,args.config)
    target.parent.mkdir(parents=True,exist_ok=True)
    # No intermediate copies: source config is immutable throughout this run.
    with target.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
            stream.write(json.dumps(data,separators=(',',':'),allow_nan=False).encode())
    print(f"Wrote {len(data['runs'])} runs to {target}; {data['seconds']:.1f} seconds; {data['execution_run_id']}",flush=True)

if __name__=='__main__':
    main()
