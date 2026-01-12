import os
import shutil
from pathlib import Path

def find_rscript() -> str:
    """
    Returns a working path to Rscript.

    Priority:
      1) ENV var RSCRIPT_PATH (absolute path)
      2) Rscript found in PATH
    """
    p = os.environ.get("RSCRIPT_PATH")
    if p:
        p = os.path.expandvars(p)
        if Path(p).exists():
            return str(Path(p))

    p2 = shutil.which("Rscript")
    if p2:
        return p2

    raise FileNotFoundError(
        "Rscript not found. Install R and ensure Rscript is on PATH, "
        "or set RSCRIPT_PATH (e.g. C:\\Program Files\\R\\R-4.5.2\\bin\\x64\\Rscript.exe)."
    )
