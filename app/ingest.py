"""File parsing: spreadsheets -> DataFrame, PDFs -> list of (page, text)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pdfplumber


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, dtype=object)
    elif suffix in (".csv", ".txt"):
        df = pd.read_csv(path, dtype=str, keep_default_na=True)
    else:
        raise ValueError(f"Unsupported table format: {suffix}")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")  # drop fully blank rows that clients love to leave in
    return df


def read_pdf_pages(path: str | Path) -> list[tuple[int, str]]:
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages.append((i, page.extract_text() or ""))
    return pages
