# tools/vecm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM


def _as_2d(df_or_none: pd.DataFrame | None) -> np.ndarray | None:
    if df_or_none is None:
        return None
    if not isinstance(df_or_none, pd.DataFrame):
        raise TypeError("exog/exog_coint must be a pandas DataFrame or None.")
    arr = np.asarray(df_or_none, dtype=float)
    if arr.ndim != 2:
        raise ValueError("exog/exog_coint must be 2D.")
    return arr


def _normalize_gammas(gamma_raw: np.ndarray, k: int, k_ar_diff: int) -> np.ndarray:
    """
    Normalize statsmodels VECMResults.gamma across versions.

    Possible shapes seen:
    - (k_ar_diff, k, k)   -> OK
    - (k, k*k_ar_diff)    -> blocks [Gamma1 | Gamma2 | ...] each (k x k)
    - (k_ar_diff, k*k)    -> flattened per lag (rare)
    """
    g = np.asarray(gamma_raw, dtype=float)

    # Case 1: already (k_ar_diff, k, k)
    if g.ndim == 3 and g.shape == (k_ar_diff, k, k):
        return g

    # Case 2: (k, k*k_ar_diff) -> split into k_ar_diff blocks
    if g.ndim == 2 and g.shape == (k, k * k_ar_diff):
        out = np.zeros((k_ar_diff, k, k), dtype=float)
        for i in range(k_ar_diff):
            out[i] = g[:, i * k : (i + 1) * k]
        return out

    # Case 3: (k_ar_diff, k*k) -> reshape each row to (k,k)
    if g.ndim == 2 and g.shape == (k_ar_diff, k * k):
        out = np.zeros((k_ar_diff, k, k), dtype=float)
        for i in range(k_ar_diff):
            out[i] = g[i].reshape(k, k)
        return out

    raise ValueError(
        f"Unsupported gamma shape {g.shape}; expected (k_ar_diff,k,k) or (k,k*k_ar_diff) or (k_ar_diff,k*k)."
    )


def _vecm_to_var_coefs(pi: np.ndarray, gammas: np.ndarray, k_ar_diff: int) -> list[np.ndarray]:
    """
    Convert VECM representation:
      Δy_t = Π y_{t-1} + Σ_{i=1}^{p-1} Γ_i Δy_{t-i} + u_t
    to VAR(p) in levels:
      y_t = Σ_{i=1}^{p} A_i y_{t-i} + u_t
    where p = k_ar_diff + 1.

    Formulas:
      A1 = I + Π + Γ1
      Ai = Γi - Γ_{i-1}  for i=2..p-1
      Ap = -Γ_{p-1}
    with Γ_i for i=1..p-1.
    """
    k = pi.shape[0]
    p = k_ar_diff + 1
    I = np.eye(k)

    if k_ar_diff < 1:
        # p=1, no gammas; then y_t = (I + Pi) y_{t-1} + u_t
        return [I + pi]

    if gammas.shape != (k_ar_diff, k, k):
        raise ValueError(f"gammas has wrong shape {gammas.shape}, expected ({k_ar_diff}, {k}, {k})")

    A: list[np.ndarray] = []

    # A1
    A1 = I + pi + gammas[0]
    A.append(A1)

    # A2..A_{p-1}
    for i in range(2, p):
        # i corresponds to Ai, gamma index i-1 and i-2 (0-based)
        if i <= p - 1:
            if i - 1 <= k_ar_diff - 1:
                Gi = gammas[i - 1]  # Γ_{i}
            else:
                Gi = np.zeros((k, k))
            Gprev = gammas[i - 2]  # Γ_{i-1}
            A.append(Gi - Gprev)

    # Ap = -Γ_{p-1}
    A.append(-gammas[k_ar_diff - 1])

    if len(A) != p:
        raise RuntimeError(f"Internal error: expected {p} VAR coefs, got {len(A)}")
    return A


def _companion_matrix(A: list[np.ndarray]) -> np.ndarray:
    """
    Build VAR(p) companion matrix F of size (k*p x k*p).
    A: list of length p, each (k x k).
    """
    p = len(A)
    k = A[0].shape[0]
    F = np.zeros((k * p, k * p), dtype=float)
    # Top block row: [A1 A2 ... Ap]
    F[:k, : k * p] = np.hstack(A)
    # Subdiagonal identity blocks
    if p > 1:
        F[k:, :-k] = np.eye(k * (p - 1))
    return F


def _ma_matrices_from_var(A: list[np.ndarray], horizon: int) -> np.ndarray:
    """
    Compute moving-average matrices Ψ_h for VAR(p), h=0..horizon.
    Returns array of shape (horizon+1, k, k).
    Recursion:
      Ψ0 = I
      Ψh = Σ_{i=1..p} A_i Ψ_{h-i}, for h>=1 (Ψ negative = 0)
    """
    p = len(A)
    k = A[0].shape[0]
    Psi = np.zeros((horizon + 1, k, k), dtype=float)
    Psi[0] = np.eye(k)

    for h in range(1, horizon + 1):
        acc = np.zeros((k, k), dtype=float)
        for i in range(1, p + 1):
            if h - i >= 0:
                acc += A[i - 1] @ Psi[h - i]
        Psi[h] = acc
    return Psi


def _orth_irf_and_fevd(A: list[np.ndarray], sigma_u: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Orthogonalized IRFs and FEVD for VAR(p).
    - IRF uses Cholesky factor P such that Sigma_u = P P'
      irf[h] = Ψ_h P  (responses to 1-s.d. orth shocks)
    - FEVD uses standard orth FEVD:
        fevd[h, i, j] = sum_{s=0..h-1} (e_i' Ψ_s P e_j)^2 / sum_{s=0..h-1} sum_{m} (e_i' Ψ_s P e_m)^2
      with horizons h=1..H.
    Returns:
      irfs: (horizon+1, k, k)  [t, response_i, shock_j]
      fevd: (horizon,   k, k)  [t(1..H), response_i, shock_j]
    """
    k = A[0].shape[0]
    sigma_u = np.asarray(sigma_u, dtype=float)
    if sigma_u.shape != (k, k):
        raise ValueError(f"sigma_u must be ({k},{k}), got {sigma_u.shape}")

    # Cholesky (lower). If not SPD, this will fail.
    P = np.linalg.cholesky(sigma_u)

    Psi = _ma_matrices_from_var(A, horizon=horizon)  # (H+1,k,k)

    irfs = np.zeros((horizon + 1, k, k), dtype=float)
    for h in range(horizon + 1):
        irfs[h] = Psi[h] @ P

    # FEVD: for horizons 1..H
    fevd = np.zeros((horizon, k, k), dtype=float)
    for h in range(1, horizon + 1):
        # accumulate contributions up to h-1
        # C_s = Psi[s] @ P, shape (k,k)
        num = np.zeros((k, k), dtype=float)  # (response_i, shock_j)
        den = np.zeros((k,), dtype=float)    # per response_i
        for s in range(h):
            Cs = Psi[s] @ P  # (k,k)
            num += Cs**2
            den += np.sum(Cs**2, axis=1)

        # divide row-wise
        for i in range(k):
            if den[i] > 0:
                fevd[h - 1, i, :] = num[i, :] / den[i]
            else:
                fevd[h - 1, i, :] = np.nan

    return irfs, fevd


def vecm_fit(
    df_endog: pd.DataFrame,
    *,
    rank: int,
    k_ar_diff: int = 2,
    deterministic: str = "ci",
    exog: pd.DataFrame | None = None,
    exog_coint: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Robust VECM fit across statsmodels versions.

    Notes:
    - Some statsmodels builds do NOT expose res.pi
    - We compute Pi = alpha @ beta.T (alpha, beta are stable)
    """
    try:
        if not isinstance(df_endog, pd.DataFrame):
            return {"ok": False, "error": "df_endog must be a pandas DataFrame."}

        X = df_endog.dropna().astype(float)
        if X.shape[0] < 30:
            return {"ok": False, "error": "Too few observations after NA drop (<30)."}

        exog_arr = _as_2d(exog) if exog is not None else None
        exog_coint_arr = _as_2d(exog_coint) if exog_coint is not None else None

        model = VECM(
            X,
            k_ar_diff=int(k_ar_diff),
            coint_rank=int(rank),
            deterministic=deterministic,
            exog=exog_arr,
            exog_coint=exog_coint_arr,
        )
        res = model.fit()

        alpha = np.asarray(getattr(res, "alpha", None), dtype=float)
        beta = np.asarray(getattr(res, "beta", None), dtype=float)
        if alpha is None or beta is None:
            return {"ok": False, "error": "VECMResults missing alpha/beta (unexpected statsmodels build)."}

        # alpha: (k x r), beta: (k x r)
        if alpha.ndim != 2 or beta.ndim != 2:
            return {"ok": False, "error": f"Unexpected alpha/beta dims: alpha {alpha.shape}, beta {beta.shape}."}
        if alpha.shape[1] != beta.shape[1]:
            return {"ok": False, "error": f"alpha and beta rank mismatch: {alpha.shape} vs {beta.shape}."}

        pi = alpha @ beta.T  # robust replacement for res.pi

        out: dict[str, Any] = {
            "ok": True,
            "n_obs": int(X.shape[0]),
            "n_vars": int(X.shape[1]),
            "columns": list(X.columns),
            "rank": int(rank),
            "k_ar_diff": int(k_ar_diff),
            "deterministic": deterministic,
            "alpha": alpha.tolist(),
            "beta": beta.tolist(),
            "pi": pi.tolist(),
            "sigma_u": np.asarray(getattr(res, "sigma_u", np.nan), dtype=float).tolist()
            if getattr(res, "sigma_u", None) is not None
            else None,
        }

        # Add text summary if available
        try:
            out["summary"] = str(res.summary())
        except Exception:
            out["summary"] = None

        return out

    except Exception as e:
        return {"ok": False, "error": "VECM fit failed", "detail": repr(e)}


def vecm_irf_fevd(
    df_endog: pd.DataFrame,
    *,
    rank: int,
    k_ar_diff: int = 2,
    deterministic: str = "ci",
    exog: pd.DataFrame | None = None,
    exog_coint: pd.DataFrame | None = None,
    horizon: int = 12,
) -> dict[str, Any]:
    """
    Compute IRF + FEVD for a VECM by converting to VAR(p) in levels
    WITHOUT relying on res.vecm_to_var() (not present in some builds).

    Output:
      - irfs: (horizon+1, k, k) [t, response_i, shock_j]
      - fevd: (horizon, k, k)   [t(1..H), response_i, shock_j]
    """
    try:
        if horizon < 1:
            return {"ok": False, "error": "horizon must be >= 1."}

        if not isinstance(df_endog, pd.DataFrame):
            return {"ok": False, "error": "df_endog must be a pandas DataFrame."}

        X = df_endog.dropna().astype(float)
        if X.shape[0] < 30:
            return {"ok": False, "error": "Too few observations after NA drop (<30)."}

        exog_arr = _as_2d(exog) if exog is not None else None
        exog_coint_arr = _as_2d(exog_coint) if exog_coint is not None else None

        model = VECM(
            X,
            k_ar_diff=int(k_ar_diff),
            coint_rank=int(rank),
            deterministic=deterministic,
            exog=exog_arr,
            exog_coint=exog_coint_arr,
        )
        res = model.fit()

        k = X.shape[1]

        alpha = np.asarray(getattr(res, "alpha", None), dtype=float)
        beta = np.asarray(getattr(res, "beta", None), dtype=float)
        if alpha is None or beta is None:
            return {"ok": False, "error": "VECMResults missing alpha/beta (unexpected statsmodels build)."}

        pi = alpha @ beta.T

        sigma_u = getattr(res, "sigma_u", None)
        if sigma_u is None:
            return {"ok": False, "error": "VECMResults.sigma_u missing; cannot compute orthogonal IRFs/FEVD."}
        sigma_u = np.asarray(sigma_u, dtype=float)

        gamma_raw = getattr(res, "gamma", None)
        if gamma_raw is None:
            return {"ok": False, "error": "VECMResults.gamma not available in this statsmodels build."}

        gammas = _normalize_gammas(gamma_raw, k=k, k_ar_diff=int(k_ar_diff))

        A = _vecm_to_var_coefs(pi=pi, gammas=gammas, k_ar_diff=int(k_ar_diff))
        irfs, fevd = _orth_irf_and_fevd(A=A, sigma_u=sigma_u, horizon=int(horizon))

        return {
            "ok": True,
            "columns": list(X.columns),
            "rank": int(rank),
            "k_ar_diff": int(k_ar_diff),
            "deterministic": deterministic,
            "var_p": int(k_ar_diff) + 1,
            "irfs": irfs.tolist(),
            "fevd": fevd.tolist(),
            "sigma_u": sigma_u.tolist(),
            "gamma_shape_raw": list(np.asarray(gamma_raw).shape),
            "gamma_shape_used": list(gammas.shape),
        }

    except np.linalg.LinAlgError as e:
        return {
            "ok": False,
            "error": "IRF/FEVD failed (Cholesky of sigma_u failed). Residual covariance not SPD.",
            "detail": repr(e),
            "hint": "Try changing variable order (Cholesky), reducing lag length, or checking data scaling.",
        }
    except Exception as e:
        return {"ok": False, "error": "IRF/FEVD failed", "detail": repr(e)}
