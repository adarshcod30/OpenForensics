"""Calibration must improve ECE without moving the ranking."""
import numpy as np
from sklearn.metrics import roc_auc_score

from openforensics.evaluation import metrics as M

rng = np.random.default_rng(0)
Y = rng.integers(0, 2, 3000)
P = np.clip(0.5 + 0.35 * (Y - 0.5) + rng.normal(0, 0.18, 3000), 1e-3, 1 - 1e-3)


def test_temperature_scaling_reduces_ece():
    t = M.fit_temperature(Y, P)
    before = M.expected_calibration_error(Y, P)
    after = M.expected_calibration_error(Y, M.apply_temperature(P, t))
    assert after < before


def test_temperature_scaling_preserves_auc():
    """A monotone transform of the score cannot change the ranking."""
    t = M.fit_temperature(Y, P)
    assert roc_auc_score(Y, P) == np.float64(roc_auc_score(Y, M.apply_temperature(P, t)))


def test_threshold_criteria_produce_valid_points():
    for c in ("youden", "f1", "target_recall", "min_fpr"):
        thr, s = M.select_threshold(Y, P, c)
        assert 0.0 < thr < 1.0
        assert 0.0 <= s["accuracy"] <= 1.0


def test_false_accusation_rate_counts_real_called_fake():
    y = np.array([1, 1, 1, 0])
    p = np.array([0.9, 0.1, 0.2, 0.1])       # two Real images scored below 0.5
    s = M.summarise(y, p, 0.5)
    assert s["real_called_fake"] == 2
    assert abs(s["false_accusation_rate"] - 2 / 3) < 1e-9


def test_risk_coverage_is_monotone_in_coverage():
    rows = M.risk_coverage(Y, P, points=10)
    assert rows[0]["coverage"] > rows[-1]["coverage"]
    assert rows[-1]["accuracy"] >= rows[0]["accuracy"] - 1e-9
