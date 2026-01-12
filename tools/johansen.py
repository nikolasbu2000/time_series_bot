from __future__ import annotations

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_test(
    df_levels: pd.DataFrame,
    det_order: int = 0,   # -1 none, 0 constant, 1 trend
    k_ar_diff: int = 1
) -> dict:
    """
    Johansen cointegration test (Trace + Max-Eigen).
    Input must be LEVELS (not differenced).
    """

    d = df_levels.copy()

    # ensure numeric
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()

    if d.shape[1] < 2:
        return {"error": "Need at least 2 variables."}
    if d.shape[0] < 50:
        return {"error": "Too few observations (<50) after alignment."}

    res = coint_johansen(
        d.values,
        det_order=int(det_order),
        k_ar_diff=int(k_ar_diff)
    )

    return {
        "n_obs": int(d.shape[0]),
        "n_vars": int(d.shape[1]),
        "columns": list(d.columns),
        "det_order": int(det_order),
        "k_ar_diff": int(k_ar_diff),

        # test statistics
        "trace_stat": [float(x) for x in res.lr1],
        "max_eig_stat": [float(x) for x in res.lr2],

        # critical values (90%, 95%, 99%)
        "trace_cv_90_95_99": res.cvt.tolist(),
        "max_eig_cv_90_95_99": res.cvm.tolist(),

        # eigen info
        "eigenvalues": [float(x) for x in res.eig],
        "eigenvectors": res.evec.tolist(),
    }
