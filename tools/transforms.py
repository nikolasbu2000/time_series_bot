from __future__ import annotations
import numpy as np
import pandas as pd

def prepare_series(df: pd.DataFrame, value_col: str, date_col: str | None = None) -> pd.Series:
    d = df.copy()
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
    s = pd.to_numeric(d[value_col], errors="coerce").dropna()
    s.name = value_col
    return s

def transform_series(s: pd.Series, use_log: bool = False, diff_k: int = 0) -> pd.Series:
    y = s.astype(float).copy()

    if use_log:
        y = y.replace([np.inf, -np.inf], np.nan).dropna()
        if (y <= 0).any():
            raise ValueError("Log-Transformation geht nur für strikt positive Werte.")
        y = np.log(y)

    if diff_k and diff_k > 0:
        y = y.diff(int(diff_k))

    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    return y
