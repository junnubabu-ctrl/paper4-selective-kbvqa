# Whole-tie selective-risk analysis

The previous risk–coverage curve split equal-confidence predictions into arbitrary partial selections. Reversing two equally scored cases could change AURC. Trapezoidal integration also omitted the interval from coverage zero to the first observed group.

This correction follows the October 7 Paper 4 manuscript's declared estimand. Sort scores descending, admit each complete equal-confidence group, and record its cumulative error rate. With coverage endpoints c_j and c_0 = 0:

    AURC = sum_j (c_j - c_(j-1)) * risk(c_j)

Every curve point is attainable by the inclusive score >= threshold rule. The first interval is included. External reported AURC values are comparable only if they use this same convention.

Inputs must be equally sized, one-dimensional arrays. Confidence must contain finite real numeric or Boolean values; strings, object arrays and complex arrays are rejected before float conversion. Correctness must be explicit real numeric or Boolean 0/1. Confidence values may be unbounded scores. Empty curves remain empty and their AURC is NaN; one case has AURC equal to its binary error. Existing Brier, ECE and selective_metrics behaviour is unchanged.

## Verification

Run from the repository root:

    PYTHONPATH=src:tests python -W error -m unittest discover -s tests -p 'test_risk_coverage_unittest.py' -v

The 15 analytic software tests passed on Python 3.12.14 / NumPy 2.3.5. They check exact integrals, complete ties, 720 input permutations, inclusive thresholds, singleton/empty inputs and malformed values. The existing test_metrics_ranges expectations are also exercised. A separate reviewer checked the mathematics and rejection of complex-score coercion.

These constructed examples verify software behaviour. They are not VQA benchmark predictions, measured accuracy, calibration results or evidence of a scientific advantage. Prior research metrics must be recomputed from preserved genuine predictions under this convention before they are reported. No historical predictions or frozen pilot inputs are changed.

This patch does not finish the primary independently calibrated likelihood comparison, whole study, human annotation or GPU execution.

