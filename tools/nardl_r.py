from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd

import os
import shutil

RSCRIPT = os.environ.get("RSCRIPT_PATH") or shutil.which("Rscript") or "Rscript"


# Adjust if your R path differs
RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe"
R_LIBS_USER = os.environ.get("R_LIBS_USER", r"C:\Users\Bublik\AppData\Local\R\win-library\4.5")


def _r_quote(name: str) -> str:
    name = str(name).replace("`", "``")
    return f"`{name}`"


def nardl_via_r(
    df: pd.DataFrame,
    y: str = "",
    x_cols: Optional[List[str]] = None,
    exog_cols: Optional[List[str]] = None,
    **kwargs,
) -> dict:
    """
    NARDL via R.
    Backward compatible wrapper:
    - accepts y_col in kwargs
    - accepts max_order (int)
    - accepts selection (AIC/BIC) etc.
    """
    y_col = kwargs.pop("y_col", None)
    if y_col is not None and str(y_col).strip():
        y = str(y_col)

    if not y or not str(y).strip():
        return {"ok": False, "error": "y (dependent variable) is missing."}

    x_cols = list(x_cols or kwargs.pop("x", []) or [])
    exog_cols = list(exog_cols or kwargs.pop("exog", []) or [])

    max_order = int(kwargs.pop("max_order", 4))
    selection = str(kwargs.pop("selection", "AIC")).upper()

    rhs = [c for c in (x_cols + exog_cols) if c is not None and str(c).strip() != ""]
    if len(rhs) == 0:
        return {"ok": False, "error": "NARDL requires at least one regressor (x_cols and/or exog_cols)."}

    if y in rhs:
        return {"ok": False, "error": "y must not appear among regressors."}

    all_cols = [y] + rhs
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        return {"ok": False, "error": f"Missing columns in dataframe: {missing}"}

    d = df[all_cols].dropna().copy()
    if len(d) < 30:
        return {"ok": False, "error": "Too few observations after alignment (<30)."}

    # formula: y ~ x1 + x2 + ...
    formula = f"{_r_quote(y)} ~ " + " + ".join(_r_quote(c) for c in rhs)

    r_script = Path(__file__).resolve().parents[1] / "r_tools" / "nardl.R"
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
        "max_order": int(max_order),
        "selection": selection,
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
            return json.loads(out)
        except Exception:
            return {
                "ok": False,
                "error": "Could not parse R JSON",
                "stdout": out[-4000:],
                "stderr": proc.stderr[-4000:],
                "debug_meta": meta,
            }
