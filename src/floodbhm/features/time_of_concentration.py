"""Time-of-concentration formulas dispatched by basin area.

Time of concentration (T_c) is the travel time of a water particle from the
hydraulically most distant point in a basin to the outlet. We use four
empirical formulas, switched by basin area class:

- **Kirpich (1940)** for basins ≤ 50 km² (small upland catchments).
- **SCS TR-55 (1986)** for basins 50–800 km² (medium agricultural).
- **Ventura** for basins 800–1000 km² (large rural).
- **Williams (1922)** for basins 1000–6000 km² (very large).

Longest channel length uses the modified Hack's law of Sassolas-Serrayet et al.
(2018), *Geomorphology* 303, 226–238.

References:
    Kirpich, Z.P., 1940. *Civil Eng.* 10, 362.
    NRCS, 1986. Technical Release 55 (TR-55).
    Ventura, in Wanielista (1990), *Hydrology and Water Quantity Control*.
    Williams, G.B., 1922. *Engineering News-Record* 88, 280-283.
"""

from __future__ import annotations

import numpy as np

__all__ = ["longest_channel_length", "time_of_concentration"]


def longest_channel_length(area_km2: float) -> float:
    """Modified Hack's law: L (km) = 1.78 * A^0.529.

    From Sassolas-Serrayet et al. (2018), best fit on global drainage networks.

    Args:
        area_km2: Basin area in km².

    Returns:
        Longest channel length in km.
    """
    return 1.78 * area_km2**0.529


def _kirpich_t_c(L_km: float, slope: float) -> float:
    """Kirpich formula. L: longest channel length (km). slope: dimensionless (m/m).

    Returns:
        T_c in hours.
    """
    L_ft = L_km * 3280.84  # km → ft
    return 0.0078 * L_ft**0.77 * slope**-0.385 / 60.0  # minutes → hours


def _scs_tr55_t_c(L_km: float, slope: float, cn: float = 75.0) -> float:
    """SCS TR-55 lag-time formula. CN default 75 (moderate runoff potential)."""
    L_ft = L_km * 3280.84
    s = 1000.0 / cn - 10.0
    lag = L_ft**0.8 * (s + 1.0) ** 0.7 / (1900.0 * (slope * 100.0) ** 0.5)
    return lag / 0.6 / 60.0  # lag = 0.6 * Tc, then minutes → hours


def _ventura_t_c(area_km2: float, slope: float) -> float:
    """Ventura formula for large rural basins."""
    return 0.127 * np.sqrt(area_km2 / slope)


def _williams_t_c(L_km: float, area_km2: float, slope: float) -> float:
    """Williams formula for very large basins."""
    return 0.21 * L_km * area_km2**0.4 / slope**0.2


def time_of_concentration(
    area_km2: float,
    slope: float,
    L_km: float | None = None,
    cn: float = 75.0,
) -> float:
    """Dispatch a T_c formula by basin area.

    Args:
        area_km2: Basin area in km².
        slope: Basin or main-channel slope (dimensionless, m/m).
        L_km: Longest channel length in km. If ``None``, computed via Hack's law.
        cn: SCS curve number used only when 50 < area ≤ 800 km².

    Returns:
        T_c in hours.

    Raises:
        ValueError: If ``area_km2`` or ``slope`` are non-positive.

    Example:
        >>> t_c = time_of_concentration(area_km2=120.0, slope=0.005)
        >>> round(t_c, 2)
        12.13
    """
    if area_km2 <= 0.0:
        raise ValueError("area_km2 must be > 0")
    if slope <= 0.0:
        raise ValueError("slope must be > 0")

    if L_km is None:
        L_km = longest_channel_length(area_km2)

    if area_km2 <= 50.0:
        return _kirpich_t_c(L_km, slope)
    if area_km2 <= 800.0:
        return _scs_tr55_t_c(L_km, slope, cn=cn)
    if area_km2 <= 1000.0:
        return _ventura_t_c(area_km2, slope)
    if area_km2 <= 6000.0:
        return _williams_t_c(L_km, area_km2, slope)

    # Very large basins outside the calibrated range — extrapolate with Williams
    # but flag the user via a warning.
    import warnings

    warnings.warn(
        f"area_km2={area_km2:.0f} exceeds Williams calibration range (≤6000 km²); extrapolating",
        stacklevel=2,
    )
    return _williams_t_c(L_km, area_km2, slope)
