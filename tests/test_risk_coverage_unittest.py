import itertools
import math
import unittest

import numpy as np

from paper4_kbvqa.evaluation.metrics import (
    aurc,
    risk_coverage_curve,
    selective_metrics,
)
from test_metrics import test_metrics_ranges


class RiskCoverageTests(unittest.TestCase):
    def test_original_range_and_threshold_expectations(self):
        test_metrics_ranges()

    def test_exact_unique_score_integral_includes_first_interval(self):
        # Risks: 1, 1/2, 1/3, 1/2; each interval has width 1/4.
        confidence = [0.9, 0.8, 0.7, 0.6]
        correct = [0, 1, 1, 0]
        coverage, risk = risk_coverage_curve(confidence, correct)
        np.testing.assert_array_equal(coverage, [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(risk, [1.0, 0.5, 1 / 3, 0.5])
        self.assertAlmostEqual(aurc(confidence, correct), 7 / 12)

    def test_analytic_tie_groups(self):
        confidence = [0.9, 0.9, 0.6, 0.6, 0.6, 0.2]
        correct = [1, 0, 1, 0, 0, 1]
        coverage, risk = risk_coverage_curve(confidence, correct)
        np.testing.assert_allclose(coverage, [1 / 3, 5 / 6, 1.0])
        np.testing.assert_allclose(risk, [0.5, 0.6, 0.5])
        # (1/3)*0.5 + (1/2)*0.6 + (1/6)*0.5 = 11/20.
        self.assertAlmostEqual(aurc(confidence, correct), 11 / 20)

    def test_all_scores_equal_yields_one_attainable_group(self):
        coverage, risk = risk_coverage_curve([0.5] * 4, [1, 0, 0, 1])
        np.testing.assert_array_equal(coverage, [1.0])
        np.testing.assert_array_equal(risk, [0.5])
        self.assertEqual(aurc([0.5] * 4, [1, 0, 0, 1]), 0.5)

    def test_tie_groups_invariant_under_all_input_permutations(self):
        confidence = np.array([0.9, 0.9, 0.6, 0.6, 0.6, 0.2])
        correct = np.array([1, 0, 1, 0, 0, 1])
        expected_coverage, expected_risk = risk_coverage_curve(confidence, correct)
        expected_aurc = aurc(confidence, correct)
        for permutation in itertools.permutations(range(len(confidence))):
            indices = list(permutation)
            coverage, risk = risk_coverage_curve(confidence[indices], correct[indices])
            np.testing.assert_array_equal(coverage, expected_coverage)
            np.testing.assert_array_equal(risk, expected_risk)
            self.assertEqual(aurc(confidence[indices], correct[indices]), expected_aurc)

    def test_curve_points_match_inclusive_thresholds_and_monotonic_coverage(self):
        confidence = np.array([0.6, 0.9, 0.2, 0.9, 0.6, 0.6])
        correct = [1, 1, 1, 0, 0, 0]
        coverage, risk = risk_coverage_curve(confidence, correct)
        self.assertTrue(np.all(np.diff(coverage) > 0))
        self.assertEqual(coverage[-1], 1.0)
        for index, threshold in enumerate(sorted(set(confidence), reverse=True)):
            result = selective_metrics(confidence, correct, threshold)
            self.assertEqual(coverage[index], result["coverage"])
            self.assertAlmostEqual(risk[index], result["selective_risk"])
        empty_selection = selective_metrics(confidence, correct, 1.0)
        self.assertEqual(empty_selection["coverage"], 0.0)
        self.assertTrue(math.isnan(empty_selection["selective_risk"]))

    def test_single_case_and_boolean_labels(self):
        for correct, expected_risk in (([1], 0.0), ([0], 1.0), ([True], 0.0), ([False], 1.0)):
            with self.subTest(correct=correct):
                coverage, risk = risk_coverage_curve([0.7], correct)
                np.testing.assert_array_equal(coverage, [1.0])
                np.testing.assert_array_equal(risk, [expected_risk])
                self.assertEqual(aurc([0.7], correct), expected_risk)

    def test_empty_curve_and_nan_integral(self):
        coverage, risk = risk_coverage_curve([], [])
        self.assertEqual(coverage.shape, (0,))
        self.assertEqual(risk.shape, (0,))
        self.assertEqual(coverage.dtype, np.dtype(float))
        self.assertEqual(risk.dtype, np.dtype(float))
        self.assertTrue(math.isnan(aurc([], [])))

    def test_confidence_scores_need_not_be_probabilities(self):
        # The ordering metric also supports finite, unbounded diagnostic scores.
        np.testing.assert_array_equal(risk_coverage_curve([2.0, -3.0], [1, 0])[0], [0.5, 1.0])

    def test_invalid_input_shapes_and_lengths(self):
        invalid = [
            (0.5, 1),
            ([[0.5]], [1]),
            ([0.5], [[1]]),
            ([0.5, 0.6], [1]),
            ([], [0]),
            ([0.5], []),
        ]
        self.assert_inputs_rejected(invalid)

    def test_nonfinite_confidence_rejected(self):
        self.assert_inputs_rejected([([value], [1]) for value in (np.nan, np.inf, -np.inf)])

    def test_complex_confidence_rejected_before_float_conversion(self):
        invalid_confidence = [
            np.array([0.5 + 9j]),
            np.array([0.5 + 0j]),
            np.array([0.5 + 9j], dtype=object),
            np.array([0.5 + 0j], dtype=object),
        ]
        self.assert_inputs_rejected([(confidence, [1]) for confidence in invalid_confidence])

    def test_string_and_object_confidence_rejected(self):
        invalid_confidence = [
            ["0.5"], ["anything"], [None],
            np.array([0.5], dtype=object),
        ]
        self.assert_inputs_rejected([(confidence, [1]) for confidence in invalid_confidence])

    def test_real_numeric_confidence_dtypes_preserved(self):
        for dtype in (np.int64, np.uint64, np.float16, np.float32, np.float64, np.bool_):
            with self.subTest(dtype=dtype):
                confidence = np.array([1, 0], dtype=dtype)
                coverage, risk = risk_coverage_curve(confidence, [1, 0])
                np.testing.assert_array_equal(coverage, [0.5, 1.0])
                np.testing.assert_array_equal(risk, [0.0, 0.5])
                self.assertEqual(aurc(confidence, [1, 0]), 0.25)

    def test_nonbinary_and_string_correctness_rejected(self):
        invalid_labels = [
            [2], [-1], [0.5], [np.nan], [np.inf], ["0"], ["1"],
            ["False"], ["anything"], [None], [1 + 0j],
            np.array([1], dtype=object),
        ]
        self.assert_inputs_rejected([([0.5], labels) for labels in invalid_labels])

    def assert_inputs_rejected(self, inputs):
        for confidence, correct in inputs:
            for function in (risk_coverage_curve, aurc):
                with self.subTest(function=function.__name__, confidence=confidence, correct=correct):
                    with self.assertRaises(ValueError):
                        function(confidence, correct)


if __name__ == "__main__":
    unittest.main()
