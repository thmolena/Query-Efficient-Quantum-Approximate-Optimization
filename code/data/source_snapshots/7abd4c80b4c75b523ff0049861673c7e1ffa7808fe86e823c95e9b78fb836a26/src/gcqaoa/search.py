"""Graph-conditioned local models with explicit finite-shot resource accounting.

All routines are independent implementations. QR, entropic structural alignment,
and confidence bounds are established ingredients, not new numerical primitives.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.special import logsumexp
from scipy.optimize import minimize
from .qaoa import normalized_shortest_path_matrix


def periods(p):
    return np.r_[np.full(p, 2*np.pi), np.full(p, np.pi/2)]


def wrap(theta):
    period = periods(len(theta)//2)
    return (np.asarray(theta) + period/2) % period - period/2


def align(theta, anchor):
    """Use exact global sign symmetry and individual periods, not basin averaging."""
    period = periods(len(theta)//2)
    choices = [anchor + (sign*np.asarray(theta)-anchor+period/2) % period-period/2
               for sign in (1, -1)]
    return min(choices, key=lambda a: np.linalg.norm(a-anchor))


def descriptor(graph):
    """Inexpensive size-independent degree and local triangle descriptors."""
    import networkx as nx
    deg = np.array([d for _, d in graph.degree()], dtype=float)
    return np.r_[np.mean(deg), np.std(deg), np.quantile(deg, [.25, .5, .75]),
                 np.mean(list(nx.clustering(graph).values()))]


def structural_score(D, E, epsilon=.05, max_iter=100, tol=1e-9):
    """Feasible entropic GW candidate, via damped fixed-point Sinkhorn steps.

    Reports unregularized squared distortion at the returned coupling. Neither
    global optimality nor distance axioms are claimed. Uniform start is part of
    the protocol; symmetry can make it uninformative on some graph families.
    """
    n, m = len(D), len(E)
    a, b = np.full(n, 1/n), np.full(m, 1/m)
    T = np.outer(a, b)
    const = (D*D) @ a[:, None] + ((E*E) @ b)[None, :]
    for iteration in range(max_iter):
        # Fixed decimal stabilization prevents roundoff-scale symmetry breaking
        # at the uniform initialization from changing basins under relabeling.
        tensor = np.round(const - 2*D @ T @ E.T, 12)
        logK = -2*tensor/epsilon
        v = np.zeros(m)
        for _ in range(500):
            u = np.log(a)-logsumexp(logK+v[None, :], axis=1)
            v = np.log(b)-logsumexp(logK+u[:, None], axis=0)
            proposal = np.exp(logK+u[:, None]+v[None, :])
            if np.max(np.abs(proposal.sum(axis=1)-a)) < 1e-12:
                break
        updated = .5*T+.5*proposal
        residual = float(np.linalg.norm(updated-T))
        T = updated
        if residual < tol:
            break
    marginal = float(max(np.max(abs(T.sum(1)-a)), np.max(abs(T.sum(0)-b))))
    if marginal > 1e-8 or not np.isfinite(T).all():
        raise ArithmeticError('Structural coupling failed marginal verification')
    score = float(np.sum((const-2*D@T@E.T)*T))
    return max(0., score), {'iterations': iteration+1, 'residual': residual,
                           'marginal_error': marginal, 'coupling': T.tolist()}


def graph_prior(graph, bank, kind='descriptor', scores=None, shuffle=None):
    """Return a reference anchor and volume-normalized bounded geometric shape.

    The anchor is the highest-weight reference, avoiding a mean of incompatible
    basins. Only aligned neighbors within Euclidean distance 1 are used to shape
    the region; the remaining weight is removed and renormalized.
    """
    start = time.perf_counter()
    descriptors = np.array([row['descriptor'] for row in bank])
    if kind == 'uniform':
        dist = np.zeros(len(bank))
    elif kind == 'descriptor':
        scale = np.std(descriptors, axis=0)
        scale[scale < .1] = .1
        dist = np.linalg.norm((descriptors-descriptor(graph))/scale, axis=1)
    elif kind in ('gw', 'local'):
        if scores is None:
            D = normalized_shortest_path_matrix(graph)
            dist = np.array([structural_score(D, np.array(row['structure']))[0] for row in bank])
        else:
            dist = np.array(scores)
    else:
        raise ValueError(kind)
    if shuffle is not None:
        dist = dist[np.asarray(shuffle)]
    tau = max(float(np.std(dist)), 1e-3)
    weights = np.exp(-(dist-dist.min())/tau)
    weights /= weights.sum()
    anchor_index = int(np.argmax(weights))
    anchor = np.array(bank[anchor_index]['theta'])
    aligned = np.array([align(row['theta'], anchor) for row in bank])
    local = np.linalg.norm(aligned-anchor, axis=1) <= 1.
    weights *= local
    weights /= weights.sum()
    diff = aligned-anchor
    covariance = .02*np.eye(len(anchor)) + (diff.T*weights)@diff
    eigen, vectors = np.linalg.eigh(covariance)
    # Bound the shape condition number by four and fix determinant to one.
    logs = np.log(eigen)
    logs -= logs.mean()
    logs = np.clip(logs, -np.log(2), np.log(2))
    logs -= logs.mean()
    B = (vectors*np.exp(.5*logs))@vectors.T
    return anchor, B, {'scores': dist.tolist(), 'weights': weights.tolist(),
                       'anchor': anchor_index, 'shape': B.tolist(),
                       'seconds': time.perf_counter()-start}


def design_points(d, rng, repair=True):
    raw = rng.normal(size=(d+1, d))
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-15)
    if repair:
        candidates = np.vstack([np.zeros((1,d)), np.eye(d), -np.eye(d), raw])
        A = np.c_[np.ones(len(candidates)), candidates]
        _, _, pivots = qr(A.T, pivoting=True, mode='economic')
        points = candidates[pivots[:d+1]]
    else:
        points = raw
    A = np.c_[np.ones(len(points)), points]
    if np.linalg.matrix_rank(A) < d+1:
        raise ArithmeticError('Rank-deficient interpolation design')
    return points, A


def fit_model(A, means, shots):
    weights = np.sqrt(np.asarray(shots))
    Q, R = qr(A*weights[:, None], mode='economic')
    coeff = solve_triangular(R, Q.T@(weights*means))
    return coeff, float(np.linalg.cond(R))


class BudgetExhausted(Exception):
    pass


@dataclass
class ShotOracle:
    circuit: object
    budget: int
    rng: object

    def __post_init__(self):
        self.shots = 0
        self.calls = 0
        self.jobs = 0
        self.records = []

    def batch(self, points, shots, stage):
        points = list(points)
        supplied = [shots]*len(points) if np.isscalar(shots) else list(shots)
        if not points or len(supplied) != len(points):
            raise ValueError('One positive shot count is required per point')
        if any(isinstance(s, (bool, np.bool_)) or not np.isfinite(s) or int(s) != s for s in supplied):
            raise ValueError('Shot counts must be finite integers')
        counts = list(map(int, supplied))
        if min(counts) < 2:
            raise ValueError('At least two shots per estimate required')
        if self.shots+sum(counts) > self.budget:
            raise BudgetExhausted
        self.jobs += 1
        result = []
        for theta, count in zip(points, counts):
            t = time.perf_counter()
            mean, variance = self.circuit.sample(theta, count, self.rng)
            self.calls += 1
            self.shots += count
            self.records.append({'call': self.calls, 'job': self.jobs, 'stage': stage,
                                 'shots': count, 'cumulative_shots': self.shots,
                                 'theta': wrap(theta).tolist(), 'mean': float(mean),
                                 'variance_of_mean': float(variance),
                                 'seconds': time.perf_counter()-t})
            result.append(mean)
        return np.array(result)


def optimize(circuit, initial, B, settings, method, seed, budget):
    """Run one frozen optimizer. Exact expectations never enter its decisions."""
    rng = np.random.default_rng(seed)
    oracle = ShotOracle(circuit, int(budget), rng)
    x = np.array(initial, dtype=float)
    start = time.perf_counter()
    decisions = []
    incumbents = [{'shots': 0, 'theta': wrap(x).tolist()}]
    d = len(x)
    stop = 'horizon'
    if method == 'cobyla':
        best = [np.inf, x.copy()]
        def objective(theta):
            y = oracle.batch([theta], settings['baseline_shots'], 'cobyla')[0]
            if y < best[0]:
                best[:] = [float(y), np.array(theta)]
                incumbents.append({'shots': oracle.shots, 'theta': wrap(theta).tolist()})
            return y
        try:
            result = minimize(objective, x, method='COBYLA', options={'rhobeg': settings['radius'],
                     'tol': .002, 'maxiter': int(budget)//settings['baseline_shots']+1})
            stop = 'converged' if result.success else 'iteration_limit'
        except BudgetExhausted:
            stop = 'budget'
        x = best[1]
    elif method == 'spsa':
        stop = 'budget'
        best = [np.inf, x.copy()]
        for iteration in range(int(budget)//(3*settings['baseline_shots'])):
            k = iteration+1
            ak = .15/(k+10)**.602
            ck = .12/k**.101
            direction = rng.choice([-1.,1.], d)
            try:
                yp, ym = oracle.batch([x+ck*direction, x-ck*direction], settings['baseline_shots'], 'spsa-gradient')
                x = wrap(x-ak*(yp-ym)/(2*ck)*direction)
                y = oracle.batch([x], settings['baseline_shots'], 'spsa-incumbent')[0]
            except BudgetExhausted:
                stop = 'budget'
                break
            if y < best[0]:
                best[:] = [float(y), x.copy()]
                incumbents.append({'shots': oracle.shots, 'theta': x.tolist()})
        x = best[1]
    else:
        radius = settings['radius']
        repair = method != 'no_repair'
        adaptive = method != 'fixed_shots'
        variable_radius = method != 'fixed_radius'
        horizon = settings['max_trials']
        looks = settings['acceptance_looks'] if adaptive else [settings['acceptance_looks'][-1]]
        # Alpha split over a prespecified finite set of cumulative looks. Points
        # may depend on the past; fresh paired streams are conditionally iid.
        logterm = np.log(4*horizon*len(looks)/settings['alpha'])
        for iteration in range(horizon):
            points, A = design_points(d, rng, repair)
            locations = [wrap(x+radius*(B@u)) for u in points]
            try:
                means = oracle.batch(locations, settings['model_shots'], 'model')
            except BudgetExhausted:
                break
            coeff, condition = fit_model(A, means, [settings['model_shots']]*(d+1))
            gradient = coeff[1:]
            pred = float(np.linalg.norm(gradient))
            if pred < 1e-12:
                stop = 'flat_model'
                break
            trial = wrap(x-radius*B@gradient/pred)
            sums = np.zeros(2)
            previous = 0
            decision = 'unresolved'
            lower = upper = None
            for cumulative in looks:
                count = cumulative-previous
                try:
                    estimates = oracle.batch([x, trial], count, 'acceptance')
                except BudgetExhausted:
                    stop = 'budget'
                    break
                sums += count*estimates
                previous = cumulative
                ahat = float((sums[0]-sums[1])/cumulative)
                bound = float(np.sqrt(logterm/(2*cumulative)))
                lower, upper = ahat-2*bound, ahat+2*bound
                if lower >= settings['eta']*pred:
                    decision = 'accepted'
                    break
                if upper < settings['eta']*pred:
                    decision = 'rejected'
                    break
            decisions.append({'iteration': iteration, 'radius': radius, 'predicted': pred,
                              'condition': condition, 'acceptance_shots_per_point': previous,
                              'lower_decrease': lower, 'upper_decrease': upper, 'decision': decision,
                              'center': wrap(x).tolist(), 'trial': trial.tolist(),
                              'shots': oracle.shots})
            if decision == 'accepted':
                x = trial
                incumbents.append({'shots': oracle.shots, 'theta': x.tolist()})
                if variable_radius:
                    radius = min(settings['max_radius'], 1.5*radius)
            elif variable_radius:
                radius = max(settings['min_radius'], .5*radius)
            if oracle.budget-oracle.shots < (d+1)*settings['model_shots']+2*looks[0]:
                stop = 'budget'
                break
    elapsed = time.perf_counter()-start
    return {'theta': wrap(x).tolist(), 'shots': oracle.shots, 'calls': oracle.calls,
            'jobs': oracle.jobs, 'seconds': elapsed, 'stop': stop, 'decisions': decisions,
            'incumbents': incumbents, 'evaluations': oracle.records}
