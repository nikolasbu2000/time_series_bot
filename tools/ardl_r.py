from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Adjust if different
RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe"
R_LIBS_USER = os.environ.get("R_LIBS_USER", r"C:\Users\Bublik\AppData\Local\R\win-library\4.5")


def _r_quote(name: str) -> str:
    name = str(name).replace("`", "``")
    return f"`{name}`"


def ardl_via_r(
    df: pd.DataFrame,
    y: str = "",
    x_cols: Optional[List[str]] = None,
    exog_cols: Optional[List[str]] = None,
    **kwargs,
) -> dict:
    """
    ARDL estimation via R package 'ARDL' (auto_ardl).

    Backward/forward compatible wrapper:
    - accepts y (positional)
    - also accepts y_col in kwargs (Streamlit code may call y_col=...)
    - accepts max_order OR max_p (aliases), selection, grid, search_type, do_bounds, bounds_case
    """
    # ---- accept both y and y_col
    y_col = kwargs.pop("y_col", None)
    if y_col is not None and str(y_col).strip():
        y = str(y_col)

    if not y or not str(y).strip():
        return {"ok": False, "error": "y (dependent variable) is missing."}

    x_cols = list(x_cols or kwargs.pop("x", []) or [])
    exog_cols = list(exog_cols or kwargs.pop("exog", []) or [])

    # ---- aliases
    max_order = kwargs.pop("max_order", None)
    if max_order is None:
        max_order = kwargs.pop("max_p", 4)
    selection = kwargs.pop("selection", "AIC")
    grid = bool(kwargs.pop("grid", False))
    search_type = kwargs.pop("search_type", "horizontal")
    do_bounds = bool(kwargs.pop("do_bounds", True))
    bounds_case = int(kwargs.pop("bounds_case", 3))

    # ignore any unknown kwargs (avoid crashes)
    _ = kwargs

    # Validate RHS
    rhs = [c for c in (x_cols + exog_cols) if c is not None and str(c).strip() != ""]
    if len(rhs) == 0:
        return {"ok": False, "error": "ARDL requires at least one regressor (x_cols and/or exog_cols)."}

    if y in rhs:
        return {"ok": False, "error": "y must not appear among regressors."}

    all_cols = [y] + rhs
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        return {"ok": False, "error": f"Missing columns in dataframe: {missing}"}

    d = df[all_cols].dropna().copy()
    if len(d) < 30:
        return {"ok": False, "error": "Too few observations after alignment (<30)."}

    # Formula
    formula = f"{_r_quote(y)} ~ " + " + ".join(_r_quote(c) for c in rhs)

    # max_order vector in exact order: [y] + regressors
    if isinstance(max_order, int):
        max_vec = [int(max_order)] * len(all_cols)
    else:
        # dict per variable name, fallback 4
        max_vec = [int(max_order.get(v, 4)) for v in all_cols]  # type: ignore[union-attr]

    # fixed_order: default exog fixed at lag 0 (dummies)
    fixed_vec = [-1] * len(all_cols)
    for i, v in enumerate(all_cols):
        if v in exog_cols:
            fixed_vec[i] = 0

    r_script = Path(__file__).resolve().parents[1] / "r_tools" / "ardl.R"
    if not r_script.exists():
        return {"ok": False, "error": f"R script not found: {r_script}"}

    env = dict(os.environ)
    env["R_LIBS_USER"] = R_LIBS_USER

    meta = {
        "formula": formula,
        "y": y,
        "x": x_cols,
        "exog": exog_cols,
        "var_order": all_cols,
        "max_order": max_vec,
        "fixed_order": fixed_vec,
        "selection": str(selection),
        "grid": bool(grid),
        "search_type": str(search_type),
        "do_bounds": bool(do_bounds),
        "bounds_case": int(bounds_case),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        data_path = tmp / "data.csv"
        meta_path = tmp / "meta.json"

        d.to_csv(data_path, index=False)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        cmd = [RSCRIPT, str(r_script), str(data_path), str(meta_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if proc.returncode != 0:
            return {
                "ok": False,
                "error": "Rscript failed",
                "stderr": proc.stderr[-4000:],
                "stdout": proc.stdout[-4000:],
                "debug_meta": meta,
            }

        out = proc.stdout.strip()
        try:
            obj = json.loads(out)
            return obj
        except Exception:
            return {
                "ok": False,
                "error": "Could not parse R JSON",
                "stdout": out[-4000:],
                "stderr": proc.stderr[-4000:],
                "debug_meta": meta,
            }
