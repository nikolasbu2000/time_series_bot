from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

# Adjust if your R installation path differs
RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe"
R_LIBS_USER = os.environ.get("R_LIBS_USER", "")


def _resolve_r_script(script_name: str) -> Path:
    """
    Your layout (as stated):
      <project_root>/r_tools/<script_name>
    NOT:
      <project_root>/tools/r_tools/...
    """
    project_root = Path(__file__).resolve().parents[1]  # .../time_series_bot
    r_script = project_root / "r_tools" / script_name
    return r_script


def _run_r_with_outfile(r_script: Path, meta: dict[str, Any], data_df: pd.DataFrame) -> dict[str, Any]:
    """
    Robust runner:
    - write meta.json + data.csv
    - call: Rscript garch.R meta.json data.csv out.json
    - parse out.json (NOT stdout), so empty stdout is fine
    """
    if not r_script.exists():
        return {"ok": False, "error": f"R script not found: {r_script}"}

    env = dict(os.environ)
    if R_LIBS_USER:
        env["R_LIBS_USER"] = R_LIBS_USER

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        meta_path = d / "meta.json"
        data_path = d / "data.csv"
        out_path = d / "out.json"

        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        data_df.to_csv(data_path, index=False)

        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(r_script), str(meta_path), str(data_path), str(out_path)],
            capture_output=True,
            text=True,
            env=env,
        )

        if proc.returncode != 0:
            return {
                "ok": False,
                "error": "Rscript failed",
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-4000:],
                "meta": meta,
            }

        if not out_path.exists():
            return {
                "ok": False,
                "error": "R did not write out.json (runner expected file output).",
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-4000:],
                "meta": meta,
            }

        raw = out_path.read_text(encoding="utf-8").strip()
        if not raw:
            return {
                "ok": False,
                "error": "out.json is empty (R produced no JSON).",
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-4000:],
                "meta": meta,
            }

        try:
            return json.loads(raw)
        except Exception as e:
            return {
                "ok": False,
                "error": "Could not parse out.json",
                "parse_error": repr(e),
                "out_raw": raw[-4000:],
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-4000:],
                "meta": meta,
            }


def ugarch_via_r(
    y: pd.Series,
    *,
    variance_model: str = "sGARCH",
    garch_p: int = 1,
    garch_q: int = 1,
    arma_p: int = 0,
    arma_q: int = 0,
    include_mean: bool = True,
    dist: str = "norm",
) -> dict[str, Any]:
    """
    Univariate GARCH via rugarch.
    Requires r_tools/garch.R to accept:
      Rscript garch.R meta.json data.csv out.json
    """
    y0 = y.dropna().astype(float)

    # rugarch needs enough observations; keep this conservative
    if len(y0) < 50:
        return {"ok": False, "error": "Series too short for GARCH (<50)."}

    r_script = _resolve_r_script("garch.R")

    meta: dict[str, Any] = {
        "mode": "ugarch",
        "variance_model": str(variance_model),
        "garch_p": int(garch_p),
        "garch_q": int(garch_q),
        "arma_p": int(arma_p),
        "arma_q": int(arma_q),
        "include_mean": bool(include_mean),
        "dist": str(dist),
    }

    data_df = pd.DataFrame({"y": y0.values})
    return _run_r_with_outfile(r_script, meta, data_df)


def dcc_mgarch_via_r(
    df: pd.DataFrame,
    cols: list[str],
    *,
    variance_model: str = "sGARCH",
    garch_p: int = 1,
    garch_q: int = 1,
    dcc_order: tuple[int, int] = (1, 1),
    dist: str = "norm",
) -> dict[str, Any]:
    """
    DCC-MGARCH via rmgarch.
    Requires r_tools/garch.R to accept:
      Rscript garch.R meta.json data.csv out.json
    """
    if len(cols) < 2:
        return {"ok": False, "error": "Need at least 2 columns for DCC-MGARCH."}

    X = df[cols].dropna().astype(float)

    # DCC fitting is heavier; enforce a slightly larger minimum
    if len(X) < 80:
        return {"ok": False, "error": "Too few rows for DCC-MGARCH (<80 after NA drop)."}

    r_script = _resolve_r_script("garch.R")

    meta: dict[str, Any] = {
        "mode": "dcc",
        "cols": list(cols),
        "variance_model": str(variance_model),
        "garch_p": int(garch_p),
        "garch_q": int(garch_q),
        "dcc_order": [int(dcc_order[0]), int(dcc_order[1])],
        "dist": str(dist),
    }

    data_df = X.reset_index(drop=True)
    return _run_r_with_outfile(r_script, meta, data_df)
