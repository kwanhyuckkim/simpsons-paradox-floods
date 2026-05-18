"""floodbhm — Hierarchical Bayesian + GP + QRF stacking for ungauged peak streamflow.

Public entry points are exposed under :mod:`floodbhm.models`, :mod:`floodbhm.features`,
and :mod:`floodbhm.eval`. See `docs/methodology.md` for the full pipeline description.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # development install without setuptools_scm
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
