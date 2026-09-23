"""Check retrospective margin diagnostics against analytic single-edge values."""
import copy
import itertools
import unittest

import numpy as np

from gcqaoa.report import (acceptance_margin_diagnostics, available_objective,
                           refinement_curves, paired_revision_interval, paired_curve_bands,
                           graph_diagnostic_interval, replay_mean_interval, fixed_power_cohorts,
                           paired_cost_saving_interval, conditional_mc_reference,
                           fresh_prediction_residuals, historical_physical_sensitivity,
                           fixed_prediction_curve)


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
    def test_old_graph_prediction_curve_is_a_fixed_lookup_not_a_zero_width_ci(self):
        rows=[{'graph':graph,'predictor':'radius_depth','shape_kind':'shape','depth':2,
               'radius':radius,'predicted_probability':probability}
              for graph in ['a','b'] for radius,probability in [(.05,.75),(.1,.25)]]
        data={'calibration_records':rows}
        np.testing.assert_array_equal(fixed_prediction_curve(data,'radius_depth',2,[.05,.1]),[.75,.25])
        rows[-1]['predicted_probability']=.4
        with self.assertRaisesRegex(ValueError,'constant across fresh targets'):
            fixed_prediction_curve(data,'radius_depth',2,[.05,.1])

    def test_physical_map_recovers_period_crossing_only_with_unique_bound(self):
        from gcqaoa.search import wrap
        center=np.array([3.12,.78])
        points=np.array([[0.,0.],[1.,0.],[0.,1.]])
        design=np.c_[np.ones(3),points]
        decision={'center':center.tolist(),'radius':.1,'condition':np.linalg.cond(design),
                  'shots':40}
        run={'decisions':[decision],
             'evaluations':[{'stage':'model','job':1,'shots':10,'cumulative_shots':10*(j+1),
                              'theta':wrap(center+.1*point).tolist()} for j,point in enumerate(points)]}
        self.assertAlmostEqual(historical_physical_sensitivity(run,np.eye(2))[0],np.sqrt(3)/.1)
        invalid=copy.deepcopy(run)
        invalid['decisions'][0]['radius']=1.
        with self.assertRaisesRegex(ValueError,'not uniquely unwrapped'):
            historical_physical_sensitivity(invalid,np.eye(2))
        invalid=copy.deepcopy(run)
        invalid['decisions'][0]['condition']*=2
        with self.assertRaisesRegex(ValueError,'saved weighted condition'):
            historical_physical_sensitivity(invalid,np.eye(2))

    def test_percentage_savings_use_ratio_of_paired_graph_means(self):
        rows=[{'graph':g,'panel':0,'method':method,'shots':shots}
              for g,stop,continuation in [('a',50,100),('b',100,400)]
              for method,shots in [('stop',stop),('continue',continuation)]]
        mean,lo,hi=paired_cost_saving_interval(rows,'stop','continue')
        self.assertEqual(mean,70.)
        self.assertEqual((lo,hi),(50.,75.))

    def test_conditional_mc_null_matches_finite_label_enumeration(self):
        # Different combined probabilities across two graphs are allowed.
        # Enumerate every allocation of 2 of 5 labeled draws to the observation.
        choices=list(itertools.combinations(range(5),2))
        residuals=[]
        for first,second in itertools.product(choices,repeat=2):
            values=[]
            for chosen,successes in [(first,2),(second,4)]:
                observed=sum(i<successes for i in chosen)
                values.append(observed/2-(successes-observed)/3)
            residuals.append(np.mean(values))
        rows=[{'graph':'a','observed_successes':1,'gaussian_successes':1},
              {'graph':'b','observed_successes':2,'gaussian_successes':2}]
        expected=np.quantile(residuals,[.025,.975],method='inverted_cdf')
        np.testing.assert_allclose(conditional_mc_reference(rows,2,3),expected)
        with self.assertRaisesRegex(ValueError,'one record per graph'):
            conditional_mc_reference(rows+rows,2,3)

    def test_zero_success_reference_is_conditional_point_mass(self):
        self.assertEqual(conditional_mc_reference([{'graph':'a','observed_successes':0,
                                                    'gaussian_successes':0}]),(0.,0.))

    def test_calibration_counts_keep_only_fresh_shaped_conditions(self):
        prediction={'id':'first','gaussian_positive_margin_probability':.5}
        row={'graph':'a','depth':2,'radius':.1,'split':'fresh','shape_kind':'shape',
             'prediction_id':'first','observed_positive_margin_fraction':.25}
        data={'config':{'proposal_repetitions':32,'gaussian_repetitions':256},
              'predictions':[prediction],'schedule_proposals':[row,dict(row,split='diagnostic'),
                                                              dict(row,shape_kind='iso')]}
        records=fresh_prediction_residuals(data)
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['observed_successes'],8)
        self.assertEqual(records[0]['gaussian_successes'],128)
        self.assertEqual(records[0]['error'],-.25)

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
