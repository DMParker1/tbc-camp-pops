# TBC Camp Populations — Link Index (Step 1)

This step crawls The Border Consortium’s “Camp Population” pages and builds a CSV index of monthly map files (PDF/JPG/PNG), parsing dates from both numeric (e.g., `2025-06`) and month-name formats (e.g., `Aug-95`, `September-1999`). No OCR/extraction yet—Step 2 will add that.

![scrape status](https://github.com/DMParker1/tbc-camp-pops/actions/workflows/scrape.yml/badge.svg)

## Output
- `data/derived/sources_index.csv` — columns: `report_date`, `source_url`, `file_name`, `origin_page`, `date_parse_method`

## How to run
1. Commit/push these files.
2. Replace `<YOUR-USER>/<YOUR-REPO>` above with your repo path.
3. Go to **Actions → scrape-and-index → Run workflow** (or wait for the cron).
