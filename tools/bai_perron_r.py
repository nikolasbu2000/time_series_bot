from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd

RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe"
R_LIBS_USER = r"C:\Users\Bublik\AppData\Local\R\win-library\4.5"

def bai_perron_via_r(
    series: pd.Series,
    model: str = "trend",     # "level" | "trend"
    max_breaks: int = 5,      # maximum m
    ic: str = "KT",           # "KT" | "BIC" | "LWZ"
    eps1: float = 0.15,       # trimming
    mode: str = "unknown",    # "unknown" | "fixed"
    k_fixed: int = 0,         # only used if mode="fixed"
) -> Dict[str, Any]:
    s = series.dropna().astype(float)
    if len(s) < 20:
        return {"error": "Series too short (<20)"}

    r_script = Path(__file__).resolve().parents[1] / "r_tools" / "bai_perron.R"
    if not r_script.exists():
        return {"error": f"R script not found: {r_script}"}

    env = dict(os.environ)
    env["R_LIBS_USER"] = R_LIBS_USER

    with tempfile.TemporaryDirectory() as d:
        y_path = Path(d) / "y.csv"
        s.to_frame("y").to_csv(y_path, index=False)

        cmd = [
            RSCRIPT,
            str(r_script),
            str(y_path),
            str(model),
            str(int(max_breaks)),
            str(ic),
            str(float(eps1)),
            str(mode),
            str(int(k_fixed)),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if proc.returncode != 0:
            return {
                "error": "Rscript failed (non-zero exit)",
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-4000:],
            }

        try:
            return json.loads(proc.stdout)
        except Exception:
            return {
                "error": "Could not parse R JSON",
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-4000:],
            }
