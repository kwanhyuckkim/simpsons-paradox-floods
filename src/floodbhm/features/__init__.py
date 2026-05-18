"""Feature engineering: peak event extraction, time of concentration, hierarchical grouping."""

from floodbhm.features.grouping import build_bhm_category, merge_small_groups_by_distance
from floodbhm.features.peak_extraction import extract_annual_peaks
from floodbhm.features.time_of_concentration import time_of_concentration

__all__ = [
    "extract_annual_peaks",
    "time_of_concentration",
    "build_bhm_category",
    "merge_small_groups_by_distance",
]
