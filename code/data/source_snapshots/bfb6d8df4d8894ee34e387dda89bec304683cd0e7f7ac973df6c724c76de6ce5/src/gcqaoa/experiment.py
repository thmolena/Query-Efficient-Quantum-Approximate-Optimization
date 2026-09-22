"""Frozen, leakage-controlled MaxCut QAOA study; run from the repository root."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
from pathlib import Path
import signal
import time
import numpy as np
import networkx as nx
import scipy
from scipy.optimize import minimize
from .qaoa import MaxCutQAOA, exact_maxcut, normalized_shortest_path_matrix, local_tv
from .search import descriptor, graph_prior, structural_score, optimize, wrap

from .provenance import PAPER_CONFIG, execution_metadata, run_id, safe_output
from .records import ExecutionRecorder, environment_details


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


def run(config, config_path=PAPER_CONFIG, *, metadata=None, recorder=None):
    """Execute a study, preserving observations when a recorder is supplied."""
    start = time.perf_counter()
    data = {'config':config, **(metadata or execution_metadata(config_path)),
            'environment':{'python':platform.python_version(), 'numpy':np.__version__,
                           'scipy':scipy.__version__, 'networkx':nx.__version__,
                           'machine':platform.machine(), 'system':platform.system(),
                           'simulation_model':'exact statevector with independent multinomial shots; no device noise',
                           'numerical_build':environment_details()},
            'instances':[], 'references':{}, 'priors':{}, 'runs':[]}
    if recorder:
        recorder.start(data)
    try:
        data['instances'] = instances(config)
        _run(config, data, recorder)
        data['seconds'] = time.perf_counter()-start
        if recorder:
            recorder.checkpoint(data)
        return data
    except BaseException as error:
        data['seconds'] = time.perf_counter()-start
        if recorder:
            recorder.abort(data, error)
        raise


def _run(config, data, recorder):
    for idx,row in enumerate(data['instances']):
        graph = graph_from(row)
        circuit = MaxCutQAOA(graph)
        row['maxcut'] = exact_maxcut(graph)[0]
        row['descriptor'] = descriptor(graph).tolist()
        row['structure'] = normalized_shortest_path_matrix(graph).tolist()
        for p in config['depths']:
            key = f"{row['id']}:p{p}"
            data['references'][key] = reference(circuit,p,config,config['seed']+10000*idx+p)
            if recorder:
                recorder.checkpoint(data)
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
            if recorder:
                recorder.checkpoint(data)
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
                        selection = {'graph':row['id'],'depth':p,'replicate':rep,
                                     'method':method,'budget':budget,'seed':seed}
                        selection['run_id'] = run_id(data['execution_run_id'], selection)
                        if recorder:
                            recorder.start_run(selection, data, x, B)
                        result=optimize(circuit,x,B,config['optimizer'],method,seed,budget,
                                        observer=recorder.event if recorder else None)
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
                        if recorder:
                            recorder.event('validation', result['validation'])
                        result.update(selection)
                        data['runs'].append(result)
                        if recorder:
                            recorder.finish_run(result)
            tests = sum(row['split']=='test' for row in data['instances'])
            print(f"experiment {idx+1}/{tests} p={p}: {len(data['runs'])} runs",flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PAPER_CONFIG)
    parser.add_argument('--output',type=Path,help='study.json.gz in a new directory under code/results; existing evidence is never replaced')
    args=parser.parse_args(argv)
    try:
        metadata=execution_metadata(args.config)
        target=safe_output(args.config,args.output,execution_id=metadata['execution_run_id'])
    except ValueError as error:
        parser.error(str(error))
    config=json.loads(args.config.read_text())
    recorder=ExecutionRecorder(target)
    previous=signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt('Execution terminated')
    signal.signal(signal.SIGTERM,terminate)
    try:
        data=run(config,args.config,metadata=metadata,recorder=recorder)
        # The interoperable bundle is derived only after all per-run records
        # have been saved. Publish a complete file without replacing evidence.
        pending=target.with_name('.study.json.gz.pending')
        try:
            with pending.open('xb') as raw:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
                    stream.write(json.dumps(data,separators=(',',':'),allow_nan=False).encode())
                raw.flush()
                os.fsync(raw.fileno())
            os.link(pending,target)
            recorder.finish(data)
        except BaseException as error:
            recorder.abort(data,error)
            raise
        finally:
            pending.unlink(missing_ok=True)
    finally:
        signal.signal(signal.SIGTERM,previous)
    print(f"Wrote {len(data['runs'])} runs to {target}; {data['seconds']:.1f} seconds; {data['execution_run_id']}",flush=True)

if __name__=='__main__':
    main()
