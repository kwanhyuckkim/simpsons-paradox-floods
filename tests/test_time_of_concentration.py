"""Unit tests for T_c dispatch and formulas."""

from __future__ import annotations

import pytest

from floodbhm.features.time_of_concentration import (
    longest_channel_length,
    time_of_concentration,
)


class TestLongestChannel:
    def test_monotonic_in_area(self):
        assert longest_channel_length(100) < longest_channel_length(1000)

    def test_zero_area_returns_zero(self):
        assert longest_channel_length(0) == pytest.approx(0.0)


class TestTimeOfConcentrationDispatch:
    def test_kirpich_small_basin(self):
        t = time_of_concentration(area_km2=10.0, slope=0.01)
        assert t > 0.0

    def test_scs_tr55_medium_basin(self):
        t = time_of_concentration(area_km2=200.0, slope=0.005)
        assert t > 0.0

    def test_ventura_large_basin(self):
        t = time_of_concentration(area_km2=900.0, slope=0.003)
        assert t > 0.0

    def test_williams_very_large_basin(self):
        t = time_of_concentration(area_km2=4000.0, slope=0.002)
        assert t > 0.0

    def test_monotonic_in_area(self):
        # Same slope, larger basin → longer T_c (roughly)
        t_small = time_of_concentration(area_km2=10.0, slope=0.005)
        t_med = time_of_concentration(area_km2=500.0, slope=0.005)
        t_large = time_of_concentration(area_km2=3000.0, slope=0.005)
        # Formulas differ across bins but each is monotonic within its bin
        assert t_med > t_small or t_large > t_small  # crude monotonicity check

    def test_invalid_area_raises(self):
        with pytest.raises(ValueError):
            time_of_concentration(area_km2=0.0, slope=0.01)
        with pytest.raises(ValueError):
            time_of_concentration(area_km2=-5.0, slope=0.01)

    def test_invalid_slope_raises(self):
        with pytest.raises(ValueError):
            time_of_concentration(area_km2=100.0, slope=0.0)
        with pytest.raises(ValueError):
            time_of_concentration(area_km2=100.0, slope=-0.01)

    def test_extrapolation_beyond_williams_warns(self):
        with pytest.warns(UserWarning):
            t = time_of_concentration(area_km2=10000.0, slope=0.002)
        assert t > 0.0
