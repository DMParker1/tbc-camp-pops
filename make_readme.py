#!/usr/bin/env python3
"""
Build README.md with:
- status badge
- last updated
- a compact table for the latest month (camp, population, category)
- links to the full CSVs
"""

from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

OUT = Path("README.md")
LONG = Path("data/derived/tbc_camp_population_long.csv")

BADGE = "![scrape status](https://github.com/DMParker1/tbc-camp-pops/actions/workflows/scrape.yml/badge.svg)"

def main():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not LONG.exists() or LONG.stat().st_size == 0:
        OUT.write_text(f"# TBC Camp Populations\n\n{BADGE}\n\n_Last updated: {ts}_\n\n"
                       "Data is not available yet.\n")
        return

    df = pd.read_csv(LONG, dtype={"camp_name":"string"})
    if df.empty or "report_date" not in df.columns:
        OUT.write_text(f"# TBC Camp Populations\n\n{BADGE}\n\n_Last updated: {ts}_\n\n"
                       "No data rows yet.\n")
        return

    # latest month
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    latest = df["report_date"].max()
    latest_str = latest.strftime("%Y-%m")
    latest_df = df[df["report_date"] == latest].copy()

    # keep tidy subset
    latest_df = latest_df[["camp_name","population","category","source_url","file_name","extract_method","parse_notes","parse_confidence"]]
    latest_df = latest_df.sort_values(["camp_name","population"], na_position="last")

    # build markdown table
    def md_table(dframe: pd.DataFrame) -> str:
        if dframe.empty:
            return "_No rows parsed for latest month._"
        cols = ["camp_name","population","category"]
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join(["---"]*len(cols)) + " |"
        body   = "\n".join([ "| " + " | ".join([str(dframe.iloc[i][c]) if pd.notna(dframe.iloc[i][c]) else "" for c in cols]) + " |"
                             for i in range(len(dframe)) ])
        return "\n".join([header, sep, body])

    table_md = md_table(latest_df)

    readme = f"""# TBC Camp Populations

{BADGE}

_Last updated: {ts}_

**Latest month:** **{latest_str}**

Below are the per-camp population counts parsed from TBC map(s) for the latest month.  
(See CSVs below for full history and parsing notes.)

{table_md}

## Data files

- `data/derived/tbc_camp_population_long.csv` — long/tidy (all months, all camps)
- `data/derived/tbc_camp_population_wide.csv` — pivot (camps as columns)

"""
    OUT.write_text(readme)

if __name__ == "__main__":
    main()
