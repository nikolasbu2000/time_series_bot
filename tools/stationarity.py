from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

@dataclass
class ADFResult:
    test_stat: float
    p_value: float
    used_lag: int
    nobs: int
    critical_values: dict
    icbest: float | None
    regression: str
    autolag: str | None

def adf_test(series: pd.Series, regression: str = "c", autolag: str | None = "AIC", maxlag: int | None = None) -> dict:
    y = series.astype(float).dropna().values
    test_stat, p_value, used_lag, nobs, crit, icbest = adfuller(
        y, regression=regression, autolag=autolag, maxlag=maxlag
    )
    res = ADFResult(
        test_stat=float(test_stat),
        p_value=float(p_value),
        used_lag=int(used_lag),
        nobs=int(nobs),
        critical_values={k: float(v) for k, v in crit.items()},
        icbest=float(icbest) if icbest is not None else None,
        regression=regression,
        autolag=autolag,
    )
    return asdict(res)

def kpss_test(series: pd.Series, regression: str = "c", nlags: str | int = "auto") -> dict:
    y = series.astype(float).dropna().values
    stat, p, lags, crit = kpss(y, regression=regression, nlags=nlags)
    return {
        "test_stat": float(stat),
        "p_value": float(p),
        "lags": int(lags),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "regression": regression,
        "nlags": nlags,
    }

def zivot_andrews_test(series: pd.Series, regression: str = "c", autolag: str = "AIC", maxlag: int | None = None) -> dict:
    """
    regression: "c" (break in intercept), "t" (break in trend), "ct" (both)
    """
    y = series.astype(float).dropna().values
    test_stat, p_value, crit, used_lag, break_index = zivot_andrews(
        y, regression=regression, autolag=autolag, maxlag=maxlag
    )
    return {
        "test_stat": float(test_stat),
        "p_value": float(p_value),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "used_lag": int(used_lag),
        "break_index": int(break_index),
        "regression": regression,
        "autolag": autolag,
        "maxlag": maxlag,
    }
