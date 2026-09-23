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


def graph_prior(graph, bank, kind='descriptor', scores=None, shuffle=None,
                tie_tolerance=1e-12):
    """Return a reference anchor and volume-normalized bounded geometric shape.

    The anchor is selected directly from scores. Scores within an absolute
    ``tie_tolerance`` of the minimum tie; the first donor in bank order wins.
    This prevents roundoff in saturated Local-TV scores from ranking donors.
    Only aligned neighbors within Euclidean distance 1 shape the region.
    """
    start = time.perf_counter()
    if not bank or not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError('A nonempty bank and nonnegative tie tolerance are required')
    descriptors = np.array([row['descriptor'] for row in bank])
    if kind == 'uniform':
        dist = np.zeros(len(bank))
    elif kind == 'descriptor':
        scale = np.std(descriptors, axis=0)
        scale[scale < .1] = .1
        dist = np.linalg.norm((descriptors-descriptor(graph))/scale, axis=1)
    elif kind in ('gw', 'local'):
        if scores is None:
            if kind == 'local':
                raise ValueError('Local-TV selection requires explicit Local-TV scores')
            D = normalized_shortest_path_matrix(graph)
            dist = np.array([structural_score(D, np.array(row['structure']))[0] for row in bank])
        else:
            dist = np.array(scores)
    else:
        raise ValueError(kind)
    if shuffle is not None:
        dist = dist[np.asarray(shuffle)]
    if dist.shape != (len(bank),) or not np.isfinite(dist).all():
        raise ValueError('One finite score is required per donor')
    tied = np.flatnonzero(dist <= dist.min() + tie_tolerance)
    anchor_index = int(tied[0])
    tau = max(float(np.std(dist)), 1e-3)
    weights = np.exp(-(dist-dist.min())/tau)
    weights /= weights.sum()
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
                       'tie_rule': 'first_in_bank_within_absolute_tolerance',
                       'tie_tolerance': float(tie_tolerance), 'tied_donors': tied.tolist(),
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
    observer: object = None

    def __post_init__(self):
        self.shots = 0
        self.calls = 0
        self.jobs = 0
        self.records = []
        self.iteration = None

    def batch(self, points, shots, stage, *, rng=None):
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
            mean, variance = self.circuit.sample(theta, count, self.rng if rng is None else rng)
            self.calls += 1
            self.shots += count
            self.records.append({'call': self.calls, 'job': self.jobs, 'stage': stage,
                                 'iteration': self.iteration,
                                 'shots': count, 'cumulative_shots': self.shots,
                                 'theta': wrap(theta).tolist(), 'mean': float(mean),
                                 'variance_of_mean': float(variance),
                                 'seconds': time.perf_counter()-t})
            if self.observer is not None:
                self.observer('evaluation', self.records[-1])
            result.append(mean)
        return np.array(result)


def optimize(circuit, initial, B, settings, method, seed, budget, observer=None):
    """Historical fixed-model-shot policy, retained for explicit comparisons.

    New refinement experiments use :func:`refine`. Exact expectations never
    enter either policy's decisions; released executions replay frozen sources.
    """
    rng = np.random.default_rng(seed)
    oracle = ShotOracle(circuit, int(budget), rng, observer)
    x = np.array(initial, dtype=float)
    start = time.perf_counter()
    decisions = []
    incumbents = [{'shots': 0, 'theta': wrap(x).tolist()}]
    if observer is not None:
        observer('incumbent', incumbents[-1])
    d = len(x)
    stop = 'horizon'
    if method == 'cobyla':
        best = [np.inf, x.copy()]
        def objective(theta):
            oracle.iteration = oracle.calls
            y = oracle.batch([theta], settings['baseline_shots'], 'cobyla')[0]
            if y < best[0]:
                best[:] = [float(y), np.array(theta)]
                incumbents.append({'shots': oracle.shots, 'theta': wrap(theta).tolist()})
                if observer is not None:
                    observer('incumbent', incumbents[-1])
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
            oracle.iteration = iteration
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
                if observer is not None:
                    observer('incumbent', incumbents[-1])
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
            oracle.iteration = iteration
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
            if observer is not None:
                observer('decision', decisions[-1])
            if decision == 'accepted':
                x = trial
                incumbents.append({'shots': oracle.shots, 'theta': x.tolist()})
                if observer is not None:
                    observer('incumbent', incumbents[-1])
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


def _pooled_endpoint(batches):
    """Unbiased pooled variance, including between-batch mean differences."""
    count, mean, m2 = 0, 0., 0.
    for batch in batches:
        n, value = batch['shots'], batch['mean']
        delta = value - mean
        m2 += n*(n-1)*batch['variance_of_mean'] + delta**2*count*n/(count+n)
        mean += delta*n/(count+n)
        count += n
    return count, float(mean), float(max(0., m2/(count-1)))


def refine(circuit, initial, B, settings, seed, budget, observer=None,
           bound='hoeffding', *, model_sampling='radius', unresolved_policy='stop'):
    """Radius-aware finite-shot refinement with explicit resolution limits.

    At radius Delta, model points receive ceil(S0*(Delta0/Delta)**2) shots
    (at least two). This preserves the leading gradient *sampling* noise scale;
    it is not a fully-linear-model accuracy guarantee. A complete model and the
    first endpoint look must be affordable before any model shots are spent.

    Fresh endpoint observations alone decide sufficient decrease. Hoeffding
    or empirical Bernstein bounds split alpha across both tails, endpoints,
    trials and the fixed cumulative-look schedule. The latter uses Theorem 4
    of Maurer and Pontil (2009), with log(8*trials*looks/alpha). Adaptive model
    sizes and proposals are measurable before their fresh endpoint samples.

    Only a certified rejection contracts the radius. Unresolved evidence stops
    with ``resolution_limited`` and leaves the incumbent unchanged. Design,
    model sampling and endpoint sampling use separate seeded random streams.

    Two optional controls isolate the interventions without changing the
    default policy. ``model_sampling='fixed'`` retains S0 at every radius.
    ``unresolved_policy='continue'`` refits a fresh model at the unchanged
    center and radius after an unresolved trial, subject to the same budget
    reservation and horizon. This is pure continuation, distinct from the
    historical policy's contraction after unresolved evidence. The same seed
    gives all four controls identical observations through their first trial.
    """
    def integer(value, minimum, name):
        if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                or int(value) != value or value < minimum):
            raise ValueError(f'{name} must be an integer >= {minimum}')
        return int(value)

    if bound not in ('hoeffding', 'bernstein'):
        raise ValueError('bound must be hoeffding or bernstein')
    if model_sampling not in ('radius', 'fixed'):
        raise ValueError('model_sampling must be radius or fixed')
    if unresolved_policy not in ('stop', 'continue'):
        raise ValueError('unresolved_policy must be stop or continue')
    budget = integer(budget, 0, 'budget')
    base_shots = integer(settings['model_shots'], 2, 'model_shots')
    horizon = integer(settings['max_trials'], 1, 'max_trials')
    looks = [integer(n, 2, 'acceptance look') for n in settings['acceptance_looks']]
    if not looks or any(b-a < 2 for a, b in zip(looks, looks[1:])):
        raise ValueError('Cumulative acceptance looks need increments of at least two')
    radius, min_radius, max_radius = map(float, (
        settings['radius'], settings['min_radius'], settings['max_radius']))
    if (not np.isfinite([radius, min_radius, max_radius]).all()
            or not 0 < min_radius <= radius <= max_radius):
        raise ValueError('Require 0 < min_radius <= radius <= max_radius')
    alpha, eta = float(settings['alpha']), float(settings['eta'])
    if not 0 < alpha < 1 or not 0 < eta < 1:
        raise ValueError('alpha and eta must lie strictly between zero and one')
    x = np.asarray(initial, dtype=float).copy()
    shape = np.asarray(B, dtype=float)
    if (x.ndim != 1 or len(x) == 0 or len(x) % 2 or not np.isfinite(x).all()
            or shape.shape != (len(x), len(x)) or not np.isfinite(shape).all()
            or np.linalg.matrix_rank(shape) != len(x)):
        raise ValueError('Require finite QAOA angles and a nonsingular matching shape')
    x = wrap(x)
    initial_radius = radius
    dimension = len(x)
    design_seed, model_seed, endpoint_seed = np.random.SeedSequence(seed).spawn(3)
    design_rng = np.random.default_rng(design_seed)
    model_rng = np.random.default_rng(model_seed)
    endpoint_rng = np.random.default_rng(endpoint_seed)
    oracle = ShotOracle(circuit, budget, model_rng, observer)
    decisions, models = [], []
    incumbents = [{'shots': 0, 'theta': x.tolist()}]
    if observer is not None:
        observer('incumbent', incumbents[-1])
    start = time.perf_counter()
    stop, stop_reason = 'horizon', 'trial_horizon'
    allocation = None
    for iteration in range(horizon):
        oracle.iteration = iteration
        model_shots = (max(2, int(np.ceil(base_shots*(initial_radius/radius)**2)))
                       if model_sampling == 'radius' else base_shots)
        model_cost = (dimension+1)*model_shots
        allocation = {'iteration': iteration, 'radius': radius,
                      'model_shots_per_point': model_shots,
                      'model_shots': model_cost, 'first_look_shots': 2*looks[0],
                      'remaining_shots': budget-oracle.shots}
        if model_cost + 2*looks[0] > budget-oracle.shots:
            stop, stop_reason = 'resolution_limited', 'model_and_first_look_unaffordable'
            break
        points, design = design_points(dimension, design_rng)
        locations = [wrap(x+radius*(shape @ point)) for point in points]
        means = oracle.batch(locations, model_shots, 'model', rng=model_rng)
        coefficient, condition = fit_model(design, means, [model_shots]*(dimension+1))
        gradient = coefficient[1:]
        predicted = float(np.linalg.norm(gradient))
        model_record = {'iteration': iteration, 'radius': radius,
                        'model_shots_per_point': model_shots, 'model_shots': model_cost,
                        'shots': oracle.shots, 'condition': condition, 'predicted': predicted,
                        'center': x.tolist(), 'design_points': points.tolist(),
                        'coefficients': coefficient.tolist()}
        models.append(model_record)
        if predicted < 1e-12:
            stop, stop_reason = 'resolution_limited', 'flat_model'
            break
        trial = wrap(x-radius*shape @ gradient/predicted)
        batches, intervals = [[], []], []
        previous = 0
        decision = 'unresolved'
        unresolved_reason = 'acceptance_look_cap'
        lower = upper = None
        for cumulative in looks:
            count = cumulative-previous
            if 2*count > budget-oracle.shots:
                unresolved_reason = 'next_acceptance_look_unaffordable'
                break
            oracle.batch([x, trial], count, 'acceptance', rng=endpoint_rng)
            for side in (0, 1):
                batches[side].append(oracle.records[-2+side])
            moments = [_pooled_endpoint(part) for part in batches]
            if bound == 'hoeffding':
                radii = [float(np.sqrt(np.log(4*horizon*len(looks)/alpha)/(2*cumulative)))]*2
            else:
                logarithm = np.log(8*horizon*len(looks)/alpha)
                radii = [float(np.sqrt(2*variance*logarithm/cumulative)
                               + 7*logarithm/(3*(cumulative-1)))
                         for _, _, variance in moments]
            decrease = moments[0][1]-moments[1][1]
            lower, upper = float(decrease-sum(radii)), float(decrease+sum(radii))
            previous = cumulative
            intervals.append({'shots_per_point': cumulative,
                              'means': [part[1] for part in moments],
                              'sample_variances': [part[2] for part in moments],
                              'radii': radii, 'lower': lower, 'upper': upper})
            if lower >= eta*predicted:
                decision = 'accepted'
                break
            if upper < eta*predicted:
                decision = 'rejected'
                break
        record = {**model_record, 'shots': oracle.shots,
                  'acceptance_shots_per_point': previous, 'acceptance_shots': 2*previous,
                  'lower_decrease': lower, 'upper_decrease': upper,
                  'decision': decision, 'trial': trial.tolist(), 'intervals': intervals,
                  'confidence_bound': bound}
        decisions.append(record)
        if observer is not None:
            observer('decision', record)
        if decision == 'accepted':
            x = trial
            incumbents.append({'shots': oracle.shots, 'theta': x.tolist()})
            if observer is not None:
                observer('incumbent', incumbents[-1])
            radius = min(max_radius, 1.5*radius)
        elif decision == 'rejected':
            if radius <= min_radius:
                stop, stop_reason = 'radius_limit', 'certified_rejection_at_min_radius'
                break
            radius = max(min_radius, .5*radius)
        elif unresolved_policy == 'stop':
            stop, stop_reason = 'resolution_limited', unresolved_reason
            break
    return {'theta': x.tolist(), 'shots': oracle.shots, 'calls': oracle.calls,
            'jobs': oracle.jobs, 'seconds': time.perf_counter()-start,
            'stop': stop, 'stop_reason': stop_reason, 'decisions': decisions,
            'incumbents': incumbents, 'evaluations': oracle.records, 'models': models,
            'policy': ('radius_adaptive' if (model_sampling, unresolved_policy) == ('radius', 'stop')
                       else f'{model_sampling}_model_{unresolved_policy}_unresolved'),
            'confidence_bound': bound,
            'final_radius': radius, 'last_allocation': allocation}
