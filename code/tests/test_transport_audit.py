import copy

import numpy as np
import pytest

from gcqaoa.transport_audit import (balanced_start, conditional_gradient,
    distortion, entropic, marginal_error, transportation, verify_summary, PROTOCOL)
from gcqaoa.search import structural_score


def test_transportation_matches_two_vertex_extremes():
    cost = np.array([[1., -2.], [-1., 3.]])
    coupling = transportation(cost)
    np.testing.assert_allclose(coupling, [[0, .5], [.5, 0]], atol=1e-10)


def test_quadratic_contraction_matches_four_index_sum():
    D = np.array([[0., .5], [.5, 0.]])
    E = np.array([[0., .3, .9], [.3, 0., .6], [.9, .6, 0.]])
    T = balanced_start(2, 3, 17)
    direct = sum((D[i,k]-E[j,l])**2*T[i,j]*T[k,l]
                 for i in range(2) for k in range(2)
                 for j in range(3) for l in range(3))
    assert abs(direct-distortion(D, E, T)) < 1e-13


def test_independent_solver_reaches_isometry_from_nonsymmetric_start():
    D = np.array([[0., 1.], [1., 0.]])
    result = conditional_gradient(D, D, balanced_start(2, 2, 19))
    assert result['score'] < 1e-12
    assert result['gap'] < 1e-10
    assert marginal_error(np.array(result['coupling'])) < 1e-10


def test_entropic_feasibility_and_stopping_flag():
    D = np.array([[0., .5], [.5, 0.]])
    E = np.array([[0., .3, .9], [.3, 0., .6], [.9, .6, 0.]])
    protocol = dict(PROTOCOL, outer_limit=1)
    result = entropic(D, E, balanced_start(2, 3, 23), .05, protocol)
    assert result['iterations'] == 1
    assert result['converged'] == (result['residual'] < protocol['outer_tolerance'])
    assert marginal_error(np.array(result['coupling'])) < 1e-8


def test_positive_scaling_agrees_with_log_domain_reference():
    D = np.array([[0., .5], [.5, 0.]])
    E = np.array([[0., .3, .9], [.3, 0., .6], [.9, .6, 0.]])
    initial = np.full((2, 3), 1/6)
    result = entropic(D, E, initial, .05, dict(PROTOCOL, outer_limit=100))
    score, reference = structural_score(D, E)
    np.testing.assert_allclose(result['coupling'], reference['coupling'], rtol=2e-8, atol=1e-10)
    assert abs(result['score']-score) < 1e-10


def summary_fixture():
    return [{'solver': 'conditional_gradient', 'depth': 1, 'pairs': 4,
             'unconverged_pairs': 0, 'infeasible_pairs': 0, 'valid_targets': 2,
             'valid_graphs': ['test_a', 'test_b'], 'changed_donors': 1,
             'selected_unconverged': 0, 'mean_deficit_pp': 0.25,
             'graph_deficits_pp': [0.1, 0.4]}]


def test_summary_accepts_only_numerical_roundoff_in_recomputed_deficits():
    expected = summary_fixture()
    saved = copy.deepcopy(expected)
    saved[0]['mean_deficit_pp'] += 4e-14
    saved[0]['graph_deficits_pp'][0] -= 4e-14
    verify_summary(expected, saved)


@pytest.mark.parametrize('field,value', [
    ('mean_deficit_pp', 0.250000001), ('graph_deficits_pp', [0.100000001, 0.4]),
    ('mean_deficit_pp', float('nan')), ('mean_deficit_pp', True),
    ('graph_deficits_pp', [0.1]), ('mean_deficit_pp', None),
    ('valid_targets', 3), ('changed_donors', 0), ('selected_unconverged', 1),
    ('valid_graphs', ['test_b', 'test_a']), ('depth', True),
])
def test_summary_rejects_material_or_discrete_changes(field, value):
    expected = summary_fixture()
    saved = copy.deepcopy(expected)
    saved[0][field] = value
    with pytest.raises(ValueError, match='summary differs'):
        verify_summary(expected, saved)


def test_summary_requires_complete_fields_and_explicit_absence_of_valid_targets():
    expected = summary_fixture()
    expected[0].update(valid_targets=0, valid_graphs=[], graph_deficits_pp=[], mean_deficit_pp=None)
    verify_summary(expected, copy.deepcopy(expected))
    for changed in ('missing_field', 'empty_row_list', 'fabricated_mean'):
        saved = copy.deepcopy(expected)
        if changed == 'missing_field':
            del saved[0]['selected_unconverged']
        elif changed == 'empty_row_list':
            saved.clear()
        else:
            saved[0]['mean_deficit_pp'] = 0.
        with pytest.raises(ValueError, match='summary'):
            verify_summary(expected, saved)
