"""Independent calibration and matched proposal/decision accounting checks."""
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gcqaoa import decision


class ConditionalCalibrationTests(unittest.TestCase):
    def test_exact_hypergeometric_reference_matches_enumeration(self):
        # Two observed and three Gaussian draws, with two combined successes.
        result = decision.conditional_null(1, 2, 1, 3)
        expected_abs = expected_square = 0.
        for k in range(3):
            weight = math.comb(2, k) * math.comb(3, 2 - k) / math.comb(5, 2)
            difference = k / 2 - (2 - k) / 3
            expected_abs += weight * abs(difference)
            expected_square += weight * difference ** 2
        self.assertAlmostEqual(result["expected_abs_error"], expected_abs)
        self.assertAlmostEqual(result["variance"], expected_square)

    def test_zero_success_null_is_degenerate_conditionally_not_a_probability_bound(self):
        result = decision.conditional_null(0, 32, 0, 256)
        self.assertEqual(result["expected_abs_error"], 0)
        self.assertEqual(result["variance"], 0)
        self.assertEqual(result["total_draws"], 288)

    def test_old_trained_baseline_does_not_use_fresh_outcomes(self):
        predictions, proposals = [], []
        for index, shape in enumerate(["iso", "shape"]):
            predictions.append({"id": index, "gaussian_proposals": [{"acceptance_margin": .1}] * 2})
            proposals.extend([
                {"prediction_id": index + 10, "graph": "old", "depth": 2, "radius": .1,
                 "shape_kind": shape, "split": "diagnostic", "observed_positive_margin_fraction": .25},
                {"prediction_id": index, "graph": "fresh", "depth": 2, "radius": .1,
                 "shape_kind": shape, "split": "fresh", "observed_positive_margin_fraction": 1.,
                 "proposals": [{"acceptance_margin": .1}] * 2}])
        records, _ = decision.calibration_analysis({"predictions": predictions, "schedule_proposals": proposals},
                                                    {"seed": 13, "null_repetitions": 8})
        baseline = [row for row in records if row["predictor"] != "gaussian"]
        self.assertEqual({row["predicted_probability"] for row in baseline}, {.25})
        self.assertEqual({row["observed_probability"] for row in baseline}, {1.})


class MatchedDecisionTests(unittest.TestCase):
    def samples(self, pairs):
        rows = []
        for center, trial in pairs:
            observations, previous = [], 0
            for n in [512, 2048, 8192, 16384]:
                observations.append({"shots_per_point": n,
                                     "batches": [{"shots": n - previous, "mean": value,
                                                  "variance_of_mean": 0.} for value in (center, trial)]})
                previous = n
            rows.append({"replays": [{"observations": observations}]})
        return rows

    def prediction(self):
        return {"id": 0, "graph": "g", "depth": 2, "radius": .1,
                "shape_kind": "shape", "scheduled_shots": 256}

    def test_failed_proposals_remain_in_gain_denominator_and_pay_model_cost(self):
        proposals = [{"predicted": .1, "true_decrease": .3, "acceptance_margin": .29},
                     {"predicted": .1, "true_decrease": -.3, "acceptance_margin": -.31}]
        config = {**decision.configuration(), "populations": {"sampled": 1}}
        records = decision.summarize_condition(self.prediction(), "sampled", proposals,
                                               self.samples([(-.5, -.8), (-.5, -.2)]), config)
        for record in records:
            self.assertEqual(record["n_proposals"], 2)
            self.assertEqual(record["acceptance_probability"], .5)
            self.assertAlmostEqual(record["mean_accepted_true_gain_pp"], 15.)
            self.assertEqual(record["mean_total_shots"], record["mean_endpoint_shots"] + 1280)

    def test_false_accepted_worsening_gain_is_signed_not_clipped(self):
        proposals = [{"predicted": .1, "true_decrease": -.2, "acceptance_margin": -.21}]
        config = {**decision.configuration(), "populations": {"sampled": 1}}
        records = decision.summarize_condition(self.prediction(), "sampled", proposals,
                                               self.samples([(-.5, -.8)]), config)
        for record in records:
            self.assertEqual(record["mean_accepted_true_gain_pp"], -20.)
            self.assertEqual(record["false_acceptances"], 1)

    def test_complete_model_and_endpoint_cost_must_be_affordable(self):
        config = decision.configuration()
        self.assertEqual(decision.available_looks(28672, 32768, config), [512, 2048, 8192, 16384])
        self.assertEqual(decision.available_looks(50000, 32768, config), [512, 2048])
        self.assertEqual(decision.available_looks(65000, 32768, config), [])
        with self.assertRaises(ValueError):
            decision.available_looks(65537, 32768, config)

    def test_full_replay_rejects_source_drift_before_executing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("changed")
            data = {"study": "decision", "schema_version": 1, "status": "complete",
                    "provenance": {"source_text": {"source.py": "original"},
                                   "source_sha256": {"source.py": hashlib.sha256(b"original").hexdigest()}}}
            data["payload_sha256"] = decision.content_hash(data)
            with patch.object(decision, "CODE_ROOT", root), patch.object(decision, "SOURCES", ("source.py",)):
                with self.assertRaisesRegex(ValueError, "requires the recorded numerical sources"):
                    decision.verify_decision(data, replay=True)


if __name__ == "__main__":
    unittest.main()
