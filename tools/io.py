from __future__ import annotations
import pandas as pd

def read_excel(uploaded_file, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def list_sheets(uploaded_file) -> list[str]:
    xls = pd.ExcelFile(uploaded_file)
    return list(xls.sheet_names)
