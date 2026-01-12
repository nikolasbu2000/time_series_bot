from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import coint


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
    trend: str = "c",          # "c", "ct", "ctt", "n"
    autolag: str | None = "aic"
) -> dict:
    """
    Engle-Granger cointegration test (two-step).
    H0: No cointegration
    """

    y = y.dropna().astype(float)
    x = x.dropna().astype(float)

    # Align series on common index
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 30:
        return {"error": "Too few overlapping observations (<30)."}

    stat, pval, crit = coint(
        df["y"],
        df["x"],
        trend=trend,
        autolag=autolag
    )

    return {
        "n_obs": int(len(df)),
        "trend": trend,
        "autolag": autolag,
        "test_statistic": float(stat),
        "p_value": float(pval),
        "critical_values": {
            "1%": float(crit[0]),
            "5%": float(crit[1]),
            "10%": float(crit[2]),
        },
        "null_hypothesis": "No cointegration",
    }
