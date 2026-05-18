"""Unit tests for floodbhm.eval.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from floodbhm.eval.metrics import (
    adjusted_r_squared,
    kling_gupta_efficiency,
    nash_sutcliffe_efficiency,
    nmpiw_iqr,
    pbias,
    picp,
    smape,
    winkler_interval_score,
)


class TestSMAPE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert smape(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        # SMAPE between [100] and [110]: 200 * 10 / (100 + 110) ≈ 9.524
        assert smape([100.0], [110.0]) == pytest.approx(9.5238, abs=1e-3)

    def test_symmetric_around_zero(self):
        # Both directions give the same SMAPE
        assert smape([5.0], [7.0]) == pytest.approx(smape([7.0], [5.0]), abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            smape([1, 2], [1, 2, 3])


class TestAdjustedR2:
    def test_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # n=5, p=2 → adj_r2 = 1 (no penalty when SSres=0)
        assert adjusted_r_squared(y, y, n_features=2) == pytest.approx(1.0, abs=1e-9)

    def test_mean_predictor_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y, y.mean())
        # SSres = SStot → R² = 0 → adj_r2 negative due to penalty
        assert adjusted_r_squared(y, y_pred, n_features=1) <= 0.0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            adjusted_r_squared([1.0, 2.0], [1.0, 2.0], n_features=2)


class TestNSE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert nash_sutcliffe_efficiency(y, y) == pytest.approx(1.0, abs=1e-9)

    def test_mean_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y, y.mean())
        assert nash_sutcliffe_efficiency(y, y_pred) == pytest.approx(0.0, abs=1e-9)


class TestKGE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert kling_gupta_efficiency(y, y) == pytest.approx(1.0, abs=1e-9)


class TestPBIAS:
    def test_no_bias(self):
        y = np.array([1.0, 2.0, 3.0])
        assert pbias(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_positive_bias(self):
        y = np.array([10.0, 10.0])
        y_pred = np.array([11.0, 11.0])
        assert pbias(y, y_pred) == pytest.approx(10.0, abs=1e-9)


class TestPredictionIntervals:
    def test_picp_full_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.5, 1.5, 2.5])
        hi = np.array([1.5, 2.5, 3.5])
        assert picp(y, lo, hi) == pytest.approx(1.0, abs=1e-9)

    def test_picp_partial_coverage(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lo = np.array([0.5, 0.5, 0.5, 0.5])
        hi = np.array([1.5, 1.5, 1.5, 1.5])
        # Only y=1.0 is inside
        assert picp(y, lo, hi) == pytest.approx(0.25, abs=1e-9)

    def test_nmpiw_iqr(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo = np.full_like(y, 0.0)
        hi = np.full_like(y, 2.0)
        # MPIW = 2.0, IQR of y = 2.0 (q1=2, q3=4) → ratio = 1.0
        assert nmpiw_iqr(y, lo, hi) == pytest.approx(1.0, abs=1e-9)

    def test_winkler_inside_interval(self):
        # When y is inside [lo, hi], score = (hi - lo)
        y = np.array([1.0])
        lo = np.array([0.0])
        hi = np.array([2.0])
        assert winkler_interval_score(y, lo, hi, alpha=0.10) == pytest.approx(2.0, abs=1e-9)

    def test_winkler_outside_above(self):
        # y above hi: score = width + (2/alpha)*(y - hi)
        y = np.array([5.0])
        lo = np.array([0.0])
        hi = np.array([2.0])
        expected = 2.0 + (2.0 / 0.10) * (5.0 - 2.0)
        assert winkler_interval_score(y, lo, hi, alpha=0.10) == pytest.approx(expected, abs=1e-9)

    def test_winkler_alpha_range(self):
        with pytest.raises(ValueError):
            winkler_interval_score([1.0], [0.0], [2.0], alpha=0.0)
        with pytest.raises(ValueError):
            winkler_interval_score([1.0], [0.0], [2.0], alpha=1.0)
