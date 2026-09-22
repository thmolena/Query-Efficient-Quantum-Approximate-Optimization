"""Check retrospective margin diagnostics against analytic single-edge values."""
import copy
import unittest

import numpy as np

from gcqaoa.report import acceptance_margin_diagnostics


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


if __name__=='__main__':
    unittest.main()
