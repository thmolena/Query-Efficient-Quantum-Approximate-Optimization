"""Reproducible sensitivity audit of structural selection on the frozen graphs.

The conditional-gradient solver is an independent implementation of the
standard GW quadratic objective with exact transportation LP subproblems.
It certifies feasibility and a first-order gap, not a global optimum.
Run with ``python -m gcqaoa.transport_audit``; use ``--verify`` to replay.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog

from .experiment import graph_from
from .provenance import CODE_ROOT, CANONICAL_DATA, digest, load_study, verify_frozen_sources
from .qaoa import MaxCutQAOA
from .records import environment_details

OUTPUT = CODE_ROOT / 'results' / 'transport.json.gz'
SOURCES = ('src/gcqaoa/transport_audit.py', 'src/gcqaoa/qaoa.py',
           'src/gcqaoa/experiment.py', 'src/gcqaoa/search.py')
PROTOCOL = {'seed': 2026091903, 'outer_limit': 1000, 'outer_tolerance': 1e-9,
            'inner_limit': 500, 'inner_tolerance': 1e-12,
            'marginal_tolerance': 1e-8, 'damping': .5, 'round_decimals': 12,
            'regularizations': [.02, .05, .1], 'random_starts': 2,
            'fw_limit': 200, 'fw_tolerance': 1e-10,
            'scaling': 'positive kernel with exponent recentering'}


def distortion(D, E, T):
    a, b = T.sum(1), T.sum(0)
    const = (D*D) @ a[:, None] + ((E*E) @ b)[None, :]
    return float(np.sum((const - 2*D@T@E.T)*T))


def marginal_error(T):
    n, m = T.shape
    return float(max(abs(T.sum(1)-1/n).max(), abs(T.sum(0)-1/m).max()))


def balanced_start(n, m, seed):
    rng = np.random.default_rng(seed)
    T = rng.uniform(.2, 1., (n, m))
    for _ in range(1000):
        T *= (1/n/T.sum(1))[:, None]
        T *= (1/m/T.sum(0))[None, :]
        if marginal_error(T) < 1e-13:
            return T
    raise ArithmeticError('Failed to construct balanced initial coupling')


def entropic(D, E, initial, epsilon, protocol=PROTOCOL):
    n, m = initial.shape
    a, b = np.full(n, 1/n), np.full(m, 1/m)
    const = (D*D) @ a[:, None] + ((E*E) @ b)[None, :]
    T = initial.copy()
    for iteration in range(protocol['outer_limit']):
        tensor = np.round(const - 2*D@T@E.T, protocol['round_decimals'])
        logK = -2*tensor/epsilon
        kernel = np.exp(logK-logK.max())
        if np.min(kernel) == 0 or not np.isfinite(kernel).all():
            raise ArithmeticError('Transport kernel outside supported numerical range')
        v = np.ones(m)
        for _ in range(protocol['inner_limit']):
            u = a/(kernel@v)
            v = b/(kernel.T@u)
            proposal = u[:, None]*kernel*v[None, :]
            if abs(proposal.sum(1)-a).max() < protocol['inner_tolerance']:
                break
        updated = (1-protocol['damping'])*T + protocol['damping']*proposal
        residual = float(np.linalg.norm(updated-T))
        T = updated
        if residual < protocol['outer_tolerance']:
            break
    error = marginal_error(T)
    if not np.isfinite(T).all():
        raise ArithmeticError('Nonfinite entropic result')
    return {'score': distortion(D, E, T), 'coupling': T.tolist(),
            'iterations': iteration+1, 'residual': residual,
            'feasible': error <= protocol['marginal_tolerance'],
            'converged': residual < protocol['outer_tolerance'] and error <= protocol['marginal_tolerance'],
            'marginal_error': error}


def transportation(gradient):
    n, m = gradient.shape
    constraints = np.vstack([np.kron(np.eye(n), np.ones((1, m))),
                             np.tile(np.eye(m), (1, n))])
    result = linprog(gradient.ravel(), A_eq=constraints,
                     b_eq=np.r_[np.full(n, 1/n), np.full(m, 1/m)],
                     bounds=(0, None), method='highs',
                     options={'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9})
    if not result.success:
        raise ArithmeticError('Transportation LP did not converge')
    return result.x.reshape(n, m)


def conditional_gradient(D, E, initial, protocol=PROTOCOL):
    T = initial.copy()
    trace = []
    for iteration in range(protocol['fw_limit']):
        gradient = -4*D@T@E.T
        direction = transportation(gradient)-T
        slope = float(np.sum(gradient*direction))
        gap = max(0., -slope)
        if gap <= protocol['fw_tolerance']:
            break
        quadratic = float(-2*np.sum((D@direction@E.T)*direction))
        candidates = [0., 1.]
        if quadratic > 0:
            candidates.append(float(np.clip(-slope/(2*quadratic), 0, 1)))
        step = min(candidates, key=lambda t: slope*t+quadratic*t*t)
        T += step*direction
        trace.append({'score': distortion(D, E, T), 'gap_before': gap, 'step': step})
    gradient = -4*D@T@E.T
    gap = max(0., float(np.sum(gradient*(T-transportation(gradient)))))
    error = marginal_error(T)
    if error > protocol['marginal_tolerance']:
        raise ArithmeticError('Infeasible conditional-gradient result')
    return {'score': distortion(D, E, T), 'coupling': T.tolist(),
            'iterations': len(trace), 'gap': gap, 'trace': trace,
            'converged': gap <= protocol['fw_tolerance'], 'feasible': True, 'marginal_error': error}


def pair_solvers(D, E, seed, protocol=PROTOCOL):
    n, m = len(D), len(E)
    starts = [np.full((n, m), 1/(n*m))]
    starts += [balanced_start(n, m, seed+i) for i in range(protocol['random_starts'])]
    results = {}
    for epsilon in protocol['regularizations']:
        results[f'entropic_{epsilon}'] = entropic(D, E, starts[0], epsilon, protocol)
    random = [entropic(D, E, T, .05, protocol) for T in starts[1:]]
    all_entropic = [results['entropic_0.05'], *random]
    results['entropic_multistart'] = {'starts': all_entropic,
        'selected': int(np.argmin([s['score'] if s['feasible'] else np.inf for s in all_entropic]))}
    fw = [conditional_gradient(D, E, T, protocol) for T in starts]
    results['conditional_gradient'] = {'starts': fw,
        'selected': int(np.argmin([s['score'] for s in fw]))}
    return results


def chosen(result):
    return result['starts'][result['selected']] if 'starts' in result else result


def summarize(data, canonical):
    banks = [r for r in canonical['instances'] if r['split'] == 'bank']
    tests = [r for r in canonical['instances'] if r['split'] == 'test']
    names = sorted(data['pairs'][0]['solvers'])
    summary = []
    for name in names:
        selected = [chosen(pair['solvers'][name]) for pair in data['pairs']]
        for p in canonical['config']['depths']:
            gaps, changes, selected_unconverged, valid_graphs = [], 0, 0, []
            for row in tests:
                records = [pair for pair in data['pairs'] if pair['test'] == row['id']]
                assert [r['bank'] for r in records] == [r['id'] for r in banks]
                # A missing feasible score invalidates the complete selector on
                # this target; do not silently discard that donor comparison.
                if any(not chosen(r['solvers'][name])['feasible'] for r in records):
                    continue
                donor = int(np.argmin([chosen(r['solvers'][name])['score'] for r in records]))
                key = f"{row['id']}:p{p}"
                old = canonical['priors'][key]['gw']['anchor']
                changes += donor != old
                selected_unconverged += not chosen(records[donor]['solvers'][name])['converged']
                theta = canonical['references'][f"{banks[donor]['id']}:p{p}"]['theta']
                objective = MaxCutQAOA(graph_from(row)).objective(theta)
                gaps.append(100*(objective-canonical['references'][key]['objective']))
                valid_graphs.append(row['id'])
            summary.append({'solver': name, 'depth': p, 'pairs': len(selected),
                            'unconverged_pairs': sum(not s['converged'] for s in selected),
                            'infeasible_pairs': sum(not s['feasible'] for s in selected),
                            'valid_targets': len(gaps), 'valid_graphs': valid_graphs,
                            'changed_donors': changes, 'selected_unconverged': selected_unconverged,
                            'mean_deficit_pp': float(np.mean(gaps)) if gaps else None,
                            'graph_deficits_pp': gaps})
    return summary


def verify_summary(expected, saved):
    """Compare selection metadata exactly and recomputed deficits numerically.

    Deficits use fresh statevector expectations, whose final rounding can differ
    across BLAS implementations even at identical package versions. The absolute
    tolerance is 1e-11 percentage points (1e-13 in normalized objective units);
    it does not apply to graph identities, counts, or solver status.
    """
    if not isinstance(saved, list) or len(saved) != len(expected):
        raise ValueError('Transport summary differs from source records')
    for fresh, recorded in zip(expected, saved):
        if not isinstance(recorded, dict) or set(fresh) != set(recorded):
            raise ValueError('Transport summary fields differ from source records')
        for key, value in fresh.items():
            actual = recorded[key]
            if key in ('mean_deficit_pp', 'graph_deficits_pp'):
                if value is None:
                    matches = actual is None
                else:
                    desired = value if isinstance(value, list) else [value]
                    observed = actual if isinstance(value, list) else [actual]
                    matches = (isinstance(observed, list) and len(observed) == len(desired)
                               and all(type(x) in (int, float) for x in observed)
                               and np.isfinite(observed).all()
                               and np.allclose(desired, observed, atol=1e-11, rtol=0))
            else:
                matches = type(actual) is type(value) and actual == value
            if not matches:
                raise ValueError('Transport summary differs from source records: ' + key)


def run(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a completed transport audit')
    canonical = load_study()
    banks = [r for r in canonical['instances'] if r['split'] == 'bank']
    tests = [r for r in canonical['instances'] if r['split'] == 'test']
    data = {'schema': 1, 'protocol': dict(PROTOCOL), 'canonical_sha256': digest(CANONICAL_DATA),
            'source_sha256': {name: digest(CODE_ROOT/name) for name in SOURCES},
            'environment': environment_details(), 'pairs': []}
    start = time.perf_counter()
    # A checkpoint inside the requested output directory preserves completed
    # solves on interruption; it is removed only after a complete bundle exists.
    pending = output.with_suffix(output.suffix+'.pending')
    if pending.exists():
        raise FileExistsError('Existing interrupted audit requires a different output path')
    for i, row in enumerate(tests):
        for j, bank in enumerate(banks):
            D, E = np.array(row['structure']), np.array(bank['structure'])
            seed = PROTOCOL['seed']+100*i+3*j
            tick = time.perf_counter()
            solvers = pair_solvers(D, E, seed)
            data['pairs'].append({'test': row['id'], 'bank': bank['id'], 'seed': seed,
                                  'solvers': solvers, 'seconds': time.perf_counter()-tick})
        with gzip.open(pending, 'wt') as stream:
            json.dump(data, stream, separators=(',', ':'), allow_nan=False)
        print(f"Transport audit {i+1}/{len(tests)} targets; {time.perf_counter()-start:.1f}s", flush=True)
    data['seconds'] = time.perf_counter()-start
    data['summary'] = summarize(data, canonical)
    with gzip.open(pending, 'wt') as stream:
        json.dump(data, stream, separators=(',', ':'), allow_nan=False)
    os.link(pending, output)
    pending.unlink()
    return data


def verify(path=OUTPUT, replay=False):
    with gzip.open(path, 'rt') as stream:
        data = json.load(stream)
    canonical = load_study()
    if data['canonical_sha256'] != digest(CANONICAL_DATA) or data['protocol'] != PROTOCOL:
        raise ValueError('Transport audit input identity mismatch')
    verify_frozen_sources(data['source_sha256'], required_paths=SOURCES)
    rows = {r['id']: r for r in canonical['instances']}
    banks = [r['id'] for r in canonical['instances'] if r['split'] == 'bank']
    tests = [r['id'] for r in canonical['instances'] if r['split'] == 'test']
    if [(r['test'], r['bank']) for r in data['pairs']] != [(t, b) for t in tests for b in banks]:
        raise ValueError('Incomplete transport comparisons')
    count = 0
    for pair in data['pairs']:
        D, E = np.array(rows[pair['test']]['structure']), np.array(rows[pair['bank']]['structure'])
        for result in pair['solvers'].values():
            starts = result.get('starts', [result])
            if 'starts' in result and result['selected'] != int(np.argmin([r['score'] if r['feasible'] else np.inf for r in starts])):
                raise ValueError('Incorrect multistart selection')
            for record in starts:
                T = np.array(record['coupling'])
                # Direct four-index calculation is independent of the efficient
                # matrix contraction used by both numerical solvers.
                direct = float(np.einsum('ikjl,ij,kl->',
                    (D[:, :, None, None]-E[None, None, :, :])**2, T, T))
                if not np.isfinite(T).all() or np.min(T) < -1e-12:
                    raise ValueError('Invalid saved coupling')
                error = marginal_error(T)
                if record['feasible'] != (error <= PROTOCOL['marginal_tolerance']):
                    raise ValueError('Incorrect feasibility status')
                if not np.isclose(error, record['marginal_error'], atol=1e-12, rtol=1e-8):
                    raise ValueError('Incorrect marginal residual')
                if not np.isclose(direct, record['score'], atol=2e-10, rtol=2e-9):
                    raise ValueError('Saved distortion does not match coupling')
                if 'gap' in record:
                    gradient = -4*D@T@E.T
                    gap = max(0., float(np.sum(gradient*(T-transportation(gradient)))))
                    if not np.isclose(gap, record['gap'], atol=2e-9, rtol=2e-8):
                        raise ValueError('Invalid conditional-gradient certificate')
                    if any(b['score'] > a['score']+1e-10 for a,b in zip(record['trace'], record['trace'][1:])):
                        raise ValueError('Conditional-gradient objective increased')
                count += 1
        if replay:
            regenerated = pair_solvers(D, E, pair['seed'], data['protocol'])
            for name in regenerated:
                if not np.isclose(chosen(regenerated[name])['score'], chosen(pair['solvers'][name])['score'], atol=2e-9, rtol=2e-8):
                    raise ValueError('Solver replay differs from saved result')
    verify_summary(summarize(data, canonical), data['summary'])
    return {'pairs': len(data['pairs']), 'couplings_checked': count, 'replayed': replay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.output, args.replay), indent=2))
    else:
        data = run(args.output)
        print(json.dumps(data['summary'], indent=2))


if __name__ == '__main__':
    main()
