"""Smoke test: every submodule imports without error."""

from __future__ import annotations


def test_top_import():
    import floodbhm

    assert hasattr(floodbhm, "__version__")


def test_eval_imports():
    from floodbhm import eval as fbev  # noqa: F401
    from floodbhm.eval import metrics, ppc, posterior_diagnostics  # noqa: F401


def test_features_imports():
    from floodbhm.features import grouping, peak_extraction, time_of_concentration  # noqa: F401


def test_models_imports():
    from floodbhm.models import bhm, gp, rfe, stacking  # noqa: F401


def test_viz_imports():
    from floodbhm.viz import simpsons  # noqa: F401
