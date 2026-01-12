from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from tools.rscript import find_rscript

R_LIBS_USER = os.environ.get("R_LIBS_USER", "")


def _as_float_series(s: pd.Series) -> pd.Series:
    return s.dropna().astype(float)


def _normalize_th_range(th_range: Any) -> Optional[list[float]]:
    if th_range is None:
        return None
    if isinstance(th_range, (list, tuple)) and len(th_range) == 2:
        return [float(th_range[0]), float(th_range[1])]
    if isinstance(th_range, str):
        parts = [p.strip() for p in th_range.split(",")]
        if len(parts) == 2:
            return [float(parts[0]), float(parts[1])]
    raise TypeError("th_range must be None, a length-2 tuple/list, or 'a,b' string.")


def apt_via_r(
    y: pd.Series,
    x: pd.Series,
    *,
    mode: str,
    model: str = "tar",
    lag: int = 1,
    thresh: float = 0.0,
    maxlag: int = 4,
    adjust: bool = True,
    th_range: Any = None,
    split: bool = False,
    which: str = "asy",
    frequency: int = 1,
    small_win: float | None = None,
) -> dict[str, Any]:
    y0 = _as_float_series(y)
    x0 = _as_float_series(x)
    df = pd.concat([y0.rename("y"), x0.rename("x")], axis=1).dropna()

    if len(df) < 40:
        return {"ok": False, "error": "Too few overlapping observations (<40)."}

    project_root = Path(__file__).resolve().parents[1]
    r_runner = project_root / "r_tools" / "apt_runner.R"
    if not r_runner.exists():
        return {"ok": False, "error": f"R runner not found: {r_runner}"}

    thr = _normalize_th_range(th_range)

    meta = {
        "mode": mode,
        "model": model,
        "lag": int(lag),
        "thresh": float(thresh),
        "maxlag": int(maxlag),
        "adjust": bool(adjust),
        "th_range": thr,
        "split": bool(split),
        "which": which,
        "frequency": int(frequency),
        "small_win": None if small_win is None else float(small_win),
        "data": {"y": df["y"].tolist(), "x": df["x"].tolist()},
    }

    env = dict(os.environ)
    if R_LIBS_USER:
        env["R_LIBS_USER"] = R_LIBS_USER

    rscript = find_rscript()

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        in_path = d / "in.json"
        out_path = d / "out.json"
        in_path.write_text(json.dumps(meta), encoding="utf-8")

        proc = subprocess.run(
            [rscript, str(r_runner), str(in_path), str(out_path)],
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
                "error": "APT runner did not write out.json",
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-4000:],
                "meta": meta,
            }

        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "ok": False,
                "error": "Could not parse out.json",
                "out_raw": out_path.read_text(encoding="utf-8")[-4000:],
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-4000:],
                "meta": meta,
            }
