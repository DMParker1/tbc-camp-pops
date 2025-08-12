#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from datetime import datetime

DERIVED = Path("data/derived")
WIDE = DERIVED / "tbc_camp_population_wide.csv"
README = Path("README.md")

def load_wide():
    if not WIDE.exists() or WIDE.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(WIDE)
    # support either "report_date" column or index
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df = df.set_index("report_date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
    return df.sort_index()

def year_end_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each year, forward-fill within the year and take the last available month per camp.
    Returns a DataFrame with years as index and camps as columns.
    """
    if df.empty:
        return df
    tmp = df.copy()
    tmp["__year__"] = tmp.index.year
    parts = []
    for y, g in tmp.groupby("__year__", dropna=True):
        g2 = g.drop(columns=["__year__"]).ffill()
        if not g2.empty:
            last = g2.tail(1)
            last.index = [y]  # year as index
            parts.append(last)
    out = pd.concat(parts) if parts else pd.DataFrame()
    # clean to integers (nullable)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    return out.sort_index()  # years ascending

def fmt_table_camps_rows(df_year: pd.DataFrame) -> str:
    """Transpose so camps are rows, format integers with thousands separators, blank for NA."""
    if df_year.empty:
        return "_No data available yet._"
    tbl = df_year.T  # camps -> rows, years -> columns
    # sort camps alphabetically for readability (optional)
    tbl = tbl.sort_index()
    tbl.index.name = "Camp"
    tbl.columns.name = None
    def fmt(x):
        try:
            return f"{int(x):,}"
        except Exception:
            return ""
    tbl = tbl.applymap(fmt)
    return tbl.to_markdown()

def main():
    df = load_wide()
    latest_month = ""
    if not df.empty and df.index.notna().any():
        try:
            latest_month = pd.to_datetime(df.index.max()).strftime("%Y-%m")
        except Exception:
            latest_month = ""

    yearly = year_end_wide(df)
    table_md = fmt_table_camps_rows(yearly)

    updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    badge = "![scrape status](https://github.com/DMParker1/tbc-camp-pops/actions/workflows/scrape.yml/badge.svg)"

    header = "# TBC Camp Populations\n\n" + badge + f"\n\n_Last updated: {updated}_\n\n"
    latest = f"**Latest month:** **{latest_month}**\n\n" if latest_month else ""
    note = (
        "### Yearly table (camps as rows)\n"
        "Values are the **latest available month** within each year (forward-filled inside the year).\n\n"
    )

    data_files = (
        "## Data files\n\n"
        "- `data/derived/tbc_camp_population_long.csv` — long/tidy (all months, all camps)\n"
        "- `data/derived/tbc_camp_population_wide.csv` — pivot (camps as columns)\n"
    )

    content = header + latest + note + table_md + "\n\n" + data_files
    README.write_text(content, encoding="utf-8")
    print("[make_readme] README.md updated.")

if __name__ == "__main__":
    main()
