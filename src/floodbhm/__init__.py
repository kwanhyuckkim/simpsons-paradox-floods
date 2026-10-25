"""floodbhm: hierarchical Bayesian + GP + QRF stacking for peak streamflow at ungauged sites.

The main pieces are in floodbhm.models, floodbhm.features and floodbhm.eval.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # development install without setuptools_scm
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
