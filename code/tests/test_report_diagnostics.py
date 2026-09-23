"""Check retrospective margin diagnostics against analytic single-edge values."""
import copy
import unittest

import numpy as np

from gcqaoa.report import (acceptance_margin_diagnostics, available_objective,
                           refinement_curves, paired_revision_interval, paired_curve_bands,
                           graph_diagnostic_interval, replay_mean_interval, fixed_power_cohorts)


class MarginDiagnosticTests(unittest.TestCase):
    def test_saved_endpoints_determine_margins_and_fixed_look_is_separate(self):
        # On a single edge these endpoints have objectives -1/2 and -1.
        # Their decrease is 1/2, while eta*q=0.496 gives margin 0.004.
        decision={'center':[0.,0.], 'trial':[np.pi/2,np.pi/8], 'predicted':4.96}
        canonical={'method':'descriptor_iso','graph':'edge',
                   'initial_objective':123.,'objective':456.,'decisions':[decision]}
        data={'instances':[{'id':'edge','n':2,'edges':[[0,1]]}],
              'config':{'optimizer':{'eta':.1,'max_trials':64,'alpha':.05,
                                   'acceptance_looks':[512,2048,8192,32768]}},
              'runs':[canonical,{**canonical,'method':'fixed_shots'},
                      {**canonical,'method':'random_iso'}]}
        original=copy.deepcopy(data)
        result=acceptance_margin_diagnostics(data)
        self.assertEqual(data,original)
        self.assertEqual(result['exact_endpoint_evaluations'],2)
        for name,trials in [('all_donor',2),('adaptive_donor',1),('fixed_shots',1)]:
            counts=result['counts'][name]
            self.assertEqual(counts['trials'],trials)
            self.assertEqual(counts['positive_true_decrease'],trials)
            self.assertEqual(counts['positive_margin'],trials)
            self.assertEqual(counts['margin_at_most_threshold'],trials)
        self.assertLess(result['adaptive_per_trial_acceptance_bound'],3.5e-6)
        # Reversing endpoints makes the true proposal harmful without changing
        # the unrelated cached initial/final diagnostic fields.
        data['runs'][0]['decisions']=[{**decision,'center':decision['trial'],
                                     'trial':decision['center']}]
        counts=acceptance_margin_diagnostics(data)['counts']['adaptive_donor']
        self.assertEqual(counts['positive_true_decrease'],0)
        self.assertEqual(counts['positive_margin'],0)


class RefinementTrajectoryTests(unittest.TestCase):
    def test_atomic_completion_and_returned_output_not_retrospective_best(self):
        run={'initial_objective':-.5,'reference_objective':-.7,
             'graph':'a','incumbents':[{'shots':0,'objective':-.5},
                                       {'shots':20,'objective':-.65},
                                       {'shots':40,'objective':-.6}]}
        self.assertEqual(available_objective(run,0),-.5)
        self.assertEqual(available_objective(run,19),-.5)
        self.assertEqual(available_objective(run,20),-.65)
        self.assertEqual(available_objective(run,39),-.65)
        self.assertEqual(available_objective(run,40),-.6)
        gain,hit=refinement_curves([run],[0,19,20,39,40],.06)
        np.testing.assert_allclose(gain,[0,0,15,15,10])
        # Target attainment is cumulative even if the available point later worsens.
        np.testing.assert_array_equal(hit,[0,0,1,1,1])

    def test_zero_shot_success_and_failed_runs_use_equal_graph_weight(self):
        a={'initial_objective':-.6,'reference_objective':-.61,'graph':'a',
           'incumbents':[{'shots':0,'objective':-.6}]}
        b={'initial_objective':-.5,'reference_objective':-.8,'graph':'b',
           'incumbents':[{'shots':0,'objective':-.5},{'shots':10,'objective':-.7}]}
        gain,hit=refinement_curves([a,a,a,b],[0,9,10,100],.02)
        np.testing.assert_allclose(gain,[0,0,10,10])
        np.testing.assert_array_equal(hit,[.5,.5,.5,.5])

    def test_pairing_requires_identical_graph_sets(self):
        base={'panel':0,'initial_objective':-.5,'objective':-.6}
        rows=[dict(base,graph='a',method='ours'),dict(base,graph='b',method='other')]
        with self.assertRaisesRegex(ValueError,'same target graphs'):
            paired_revision_interval(rows,'ours','other')

    def test_paired_effect_averages_repetitions_before_graph_bootstrap(self):
        base={'panel':0,'initial_objective':-.5}
        rows=[dict(base,graph='a',method='ours',objective=-.7),
              dict(base,graph='a',method='ours',objective=-.7),
              dict(base,graph='a',method='other',objective=-.6),
              dict(base,graph='b',method='ours',objective=-.5),
              dict(base,graph='b',method='other',objective=-.6)]
        mean,lo,hi=paired_revision_interval(rows,'ours','other')
        self.assertAlmostEqual(mean,0.)
        self.assertAlmostEqual(lo,-10.)
        self.assertAlmostEqual(hi,10.)


class MechanismReportTests(unittest.TestCase):
    def test_rms_is_root_of_graph_weighted_squared_errors(self):
        rows=[{'graph':'a','error':1.},{'graph':'a','error':3.},
              {'graph':'b','error':2.}]
        mean,lo,hi=graph_diagnostic_interval(rows,lambda r:r['error'],rms=True)
        self.assertAlmostEqual(mean,np.sqrt(4.5))
        self.assertAlmostEqual(lo,2.)
        self.assertAlmostEqual(hi,np.sqrt(5.))

    def test_paired_bands_keep_entire_graph_trajectory_together(self):
        rows=[]
        for graph,ending in [('a',-.7),('b',-.5)]:
            rows.append({'graph':graph,'panel':0,'method':'revised','initial_objective':-.5,
                         'incumbents':[{'shots':0,'objective':-.5},{'shots':10,'objective':ending}]})
            rows.append({'graph':graph,'panel':0,'method':'original','initial_objective':-.5,
                         'incumbents':[{'shots':0,'objective':-.5},{'shots':10,'objective':-.6}]})
        mean,lo,hi=paired_curve_bands(rows,[0,9,10,20],'revised','original')
        np.testing.assert_allclose(mean,[0,0,0,0],atol=1e-12)
        np.testing.assert_allclose(lo,[0,0,-10,-10])
        np.testing.assert_allclose(hi,[0,0,10,10])

    def test_zero_empirical_frequency_does_not_give_zero_upper_bound(self):
        mean,lo,hi=replay_mean_interval([{'repeats':128,'acceptance_probability':0.}],
                                       'acceptance_probability',32768)
        self.assertEqual((mean,lo),(0.,0.))
        self.assertGreater(hi,.0231)
        self.assertLess(hi,1.)

    def test_power_cohort_cannot_change_with_budget(self):
        rows=[{'proposal_id':1,'endpoint_pair_budget':1024},
              {'proposal_id':1,'endpoint_pair_budget':4096},
              {'proposal_id':2,'endpoint_pair_budget':4096}]
        with self.assertRaisesRegex(ValueError,'identical fixed proposal'):
            fixed_power_cohorts(rows)



if __name__=='__main__':
    unittest.main()
