from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def stl_decompose(
    s: pd.Series,
    *,
    period: int,
    robust: bool = False,
) -> dict[str, Any]:
    """
    STL decomposition using statsmodels.

    Returns JSON-friendly dict with arrays (lists) so Streamlit can display easily.
    """
    s0 = s.dropna().astype(float)
    if len(s0) < max(2 * period + 5, 20):
        return {
            "ok": False,
            "error": f"Series too short for STL (need roughly >= 2*period). len={len(s0)}, period={period}",
        }

    try:
        from statsmodels.tsa.seasonal import STL
    except Exception as e:
        return {"ok": False, "error": f"statsmodels STL import failed: {e}"}

    try:
        res = STL(s0, period=int(period), robust=bool(robust)).fit()

        out = {
            "ok": True,
            "period": int(period),
            "robust": bool(robust),
            "index": [str(x) for x in s0.index],  # safe for JSON
            "observed": s0.tolist(),
            "trend": pd.Series(res.trend, index=s0.index).tolist(),
            "seasonal": pd.Series(res.seasonal, index=s0.index).tolist(),
            "resid": pd.Series(res.resid, index=s0.index).tolist(),
        }
        return out

    except Exception as e:
        return {"ok": False, "error": f"STL failed: {e}"}
