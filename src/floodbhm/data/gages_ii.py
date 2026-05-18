"""GAGES-II watershed boundary + basin-characteristic loaders.

TODO: implement the boundary shapefile reader + the
``basinchar_and_report_sept_2011/`` conterm CSV merge that selects the 18
features used by the BHM. The cached parquet on Zenodo skips this step for
standard reproduction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_gages_ii_basinchar"]


GAGES_II_FEATURES_18 = [
    "BAS_COMPACTNESS",
    "STREAMS_KM_SQ_KM",
    "MAINSTEM_SINUOUSITY",
    "BFI_AVE",
    "ELEV_MEAN_M_BASIN",
    "SLOPE_PCT",
    "LAT_GAGE",
    "LNG_GAGE",
    "PRECIP_SEAS_IND",
    "RFACT",
    "AWCAVE",
    "PERMAVE",
    "ROCKDEPAVE",
    "CLAYAVE",
    "SILTAVE",
    "SANDAVE",
    "OMAVE",
    "AREA",
]


def load_gages_ii_basinchar(
    basinchar_dir: Path,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Load and merge GAGES-II conterm CSV files.

    Args:
        basinchar_dir: Directory containing ``conterm*.txt`` files.
        features: Subset of columns to keep. Defaults to the 18 used in the paper.

    Returns:
        DataFrame indexed by ``GAGE_ID`` with the requested feature columns.
    """
    features = features or GAGES_II_FEATURES_18
    paths = sorted(Path(basinchar_dir).glob("conterm*.txt"))
    if not paths:
        raise FileNotFoundError(f"No conterm*.txt files found under {basinchar_dir}")

    dfs = []
    for path in paths:
        df = pd.read_csv(path, converters={"STAID": str}, encoding="iso-8859-1")
        if "STAID" in df.columns:
            df = df.rename(columns={"STAID": "GAGE_ID"})
        dfs.append(df)

    merged = pd.concat(dfs, axis=1).dropna(axis=1, how="all")
    # If column appears in multiple files keep first
    merged = merged.loc[:, ~merged.columns.duplicated()]
    cols = ["GAGE_ID"] + [c for c in features if c in merged.columns]
    return merged[cols].set_index("GAGE_ID")
