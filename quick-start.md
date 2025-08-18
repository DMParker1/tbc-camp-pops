# Scrape & Extract — Action Quick Start

A practical guide to running the **scrape-and-index** workflow and knowing which knobs to turn for fast iteration vs. full runs.

**Workflow file:** `.github/workflows/scrape-and-index.yml`  
**Scripts:**  
- `tbc_scraper.py` → crawls monthly sources and builds `data/derived/sources_index.csv`  
- `tbc_extract.py` → parses PDFs into `data/derived/tbc_camp_population_*.csv`  
- `make_readme.py` → updates the yearly table in `README.md`

---

## Most common runs (presets)

### 1) Fast dev run (small window, reuse index)
Use this to validate changes quickly without waiting on a full crawl.

**GitHub UI:** Actions → **scrape-and-index** → **Run workflow**  
Set:
- `extract_since`: `2011-10-01`
- `extract_until`: `2011-12-31`
- `process_order`: `oldest`
- `extract_max_files`: `12`
- `skip_crawl`: **true**
- `seed_only`: **true**
- `stop_after_found`: **true**
- `max_pages`: `2`

**CLI (gh):**
~~~bash
gh workflow run scrape-and-index \
  -f extract_since=2011-10-01 \
  -f extract_until=2011-12-31 \
  -f process_order=oldest \
  -f extract_max_files=12 \
  -f skip_crawl=true \
  -f seed_only=true \
  -f stop_after_found=true \
  -f scrape_since=1992-01-01 \
  -f max_pages=2
~~~

---

### 2) Extract-only (no new crawl)
Re-use the existing `sources_index.csv` and just re-run parsing.
- `skip_crawl`: **true**  
- Keep your preferred `extract_*` inputs.

---

### 3) Crawl-only (refresh index, skip parsing)
Update the URL index but do not parse PDFs this run.
- Set `extract_max_files`: `0` (extract step will no-op; README still builds)

---

### 4) Full backfill (end-to-end, oldest first)
Let the crawler find sources and parse everything.
- `extract_since`: *(blank)*
- `extract_until`: *(blank)*
- `process_order`: `oldest`
- `skip_crawl`: **false**
- `seed_only`: **false** (or leave **true** to be conservative)
- `stop_after_found`: **true**
- `max_pages`: `2000` (default)

> Tip: for very long runs, start with a mid-size `max_pages` (e.g., 200–400) to confirm crawl coverage, then raise.

---

## Inputs explained (with examples)

These are the **workflow_dispatch** inputs, how they map to env vars, and what they do.

### Extraction window & order
- **`extract_since`** (string, default `""`) → **EXTRACT_SINCE**  
  Lower bound (inclusive) for the extractor in `YYYY-MM-DD`.  
  *Example:* `2011-10-01` processes Oct 2011 onward.

- **`extract_until`** (string, default `""`) → **EXTRACT_UNTIL**  
  Upper bound (inclusive) for the extractor in `YYYY-MM-DD`.  
  *Example:* `2011-12-31` stops at Dec 2011.

- **`process_order`** (choice, default `oldest`) → **PROCESS_ORDER**  
  Order of files after date filtering: `oldest` or `newest`.  
  *Example:* Use `oldest` when backfilling; use `newest` when you only want the latest few months first.

- **`extract_max_files`** (string, default `"250"`) → **EXTRACT_MAX_FILES**  
  Cap how many files the extractor will process in this run. `"0"` skips extraction entirely.  
  *Examples:*  
  • Debug: `2` (just a couple PDFs)  
  • Small QA: `12` (a few months)  
  • Full: `250+`

### Crawl controls (building `sources_index.csv`)
- **`skip_crawl`** (boolean, default **true**) → **SKIP_CRAWL**  
  If **true**, reuse existing `data/derived/sources_index.csv`. Choose this for fast iteration on the extractor.

- **`seed_only`** (boolean, default **true**) → **SEED_ONLY**  
  If **true**, the crawler only hits seed/search pages (doesn’t follow many deep links). Faster, but may miss some alternate mirrors.

- **`stop_after_found`** (boolean, default **true**) → **STOP_AFTER_FOUND**  
  If **true**, after the first **OK** candidate link per month is found, the crawler stops probing more links for that month. Great for speed; set to **false** if you want to find alternates or handle flaky mirrors.

- **`scrape_since`** (string, default `1992-01-01`) → **SCRAPE_SINCE**  
  Lower bound (inclusive) for the **crawler’s candidate months**. This does **not** force the extractor to run—only which months the index will include.

- **`scrape_until`** (string, default `""`) → **SCRAPE_UNTIL**  
  Upper bound (inclusive) for the **crawler’s candidate months**. Blank = today.

- **`max_pages`** (string, default `"2000"`) → **MAX_PAGES** — **Crawler page budget**  
  The **maximum number of web pages** the crawler will fetch **during this run**. This budget includes:
  - seed/search pages (e.g., archive landing pages)
  - if `seed_only=false`, deeper archive/detail pages it follows from those seeds

  What different values mean:
  - **`0`** → *Disable crawl entirely.* Use this to ensure no web requests are made (you’ll rely on an existing `sources_index.csv`).
  - **1–5** → *Lightning sanity checks.* Typically just the first search page or two. Great for debugging the crawler without heavy network use.
  - **50–400** → *Moderate breadth.* Reasonable coverage across many years while keeping runs short.
  - **1000–2000** → *Deep sweeps.* Full archive passes.
  
  How it interacts with other knobs:
  - With **`seed_only=true`**, the budget is spent almost entirely on search pages → high breadth, low depth.
  - With **`seed_only=false`**, budget is also spent following candidate links into month/detail pages → more depth.
  - With **`stop_after_found=true`**, you conserve budget by not probing additional candidates once the first good link is found for a month.

  *Example:* Use `max_pages=2`, `seed_only=true`, `stop_after_found=true` for a ~5–10 second “is the crawler wiring okay?” pass.

> **Rule of thumb:** If the crawl returns fewer months than expected, raise `max_pages` and/or set `seed_only=false` for a pass, then switch back once coverage looks good.

---

## What each job step does (with how it reads inputs)

1) **Crawl + index** — `python -u tbc_scraper.py`  
   **Env used:** `SKIP_CRAWL`, `SEED_ONLY`, `MAX_PAGES`, `STOP_AFTER_FOUND`, `PROCESS_ORDER`, `SCRAPE_SINCE`, `SCRAPE_UNTIL`  
   **Output:** `data/derived/sources_index.csv`  
   **When to tweak:**  
   - Turn **skip_crawl=true** while debugging the extractor.  
   - Lower **max_pages** (e.g., 2–10) for a quick “is it working?” pass.  
   - Keep **stop_after_found=true** to avoid overspending the budget.  
   **Example:** Refresh just the index for early 2000s with a tight budget:  
   `seed_only=true, stop_after_found=true, max_pages=50, scrape_since=2000-01-01, scrape_until=2005-12-31`

2) **Extract populations** — `python -u tbc_extract.py`  
   **Env used:** `EXTRACT_SINCE`, `EXTRACT_UNTIL`, `EXTRACT_MAX_FILES`, `PROCESS_ORDER`  
   **Hardcoded defaults (from workflow):** `RESUME_MODE=true`, `OCR_DPI=200`, `OCR_PSM=6`  
   **Note on OCR:** `DISABLE_OCR` is **not set** in the workflow, so the extractor’s own default applies (fast path = OCR off).  
   To temporarily enable OCR for a small window, add this to the **Extract** step’s `env:` while testing:  
   ~~~yaml
   DISABLE_OCR: "false"   # enable OCR for this run only
   ~~~
   **Outputs:**  
   - `data/derived/tbc_camp_population_long.csv` — tidy/long (all rows)  
   - `data/derived/tbc_camp_population_wide.csv` — **dates as index**, camps as columns  
   - `data/derived/tbc_camp_population_wide_dual.csv` — camps as rows; columns like `YYYY-MM-DD|tbbc` & `YYYY-MM-DD|unhcr`
   
   **Typical examples:**  
   - *Small QA (12 files, Oct–Dec 2011, oldest first):*  
     `extract_since=2011-10-01`, `extract_until=2011-12-31`, `extract_max_files=12`, `process_order=oldest`  
   - *Latest-first spot check (few files):*  
     `extract_since=` *(blank)*, `extract_until=` *(blank)*, `process_order=newest`, `extract_max_files=6`  
   - *Extract-only pass (no crawl):*  
     `skip_crawl=true`, set your `extract_*` window, leave everything else default

3) **Build README table** — `python -u make_readme.py`  
   Reads **WIDE** and writes the yearly “camps as rows” table.  
   If the table looks wrong (e.g., “1970” header), your **WIDE** shape likely flipped—confirm the first column is `report_date`.

4) **Commit outputs**  
   Commits updated CSVs + README to the branch.

---

## Quick verification (in GitHub web UI)

- **WIDE shape:** open `data/derived/tbc_camp_population_wide.csv` → **Raw**.  
  Header starts with `report_date`, followed by **camp names**. First data row begins with a date like `2011-10-01`.

- **Dual columns present:** open `data/derived/tbc_camp_population_wide_dual.csv` → **Raw**.  
  Find `2011-10-01|tbbc` and `2011-10-01|unhcr` (paired for each date).

- **README updated:** open `README.md`.  
  Yearly table shows **Camp** in the first column and **years** as headers.

---

## Troubleshooting

- **Nothing in LONG / placeholders with `no_rows_parsed`**  
  Likely a geometry bug or too-narrow crawl. Try a tiny run with OCR enabled just for that window: set `DISABLE_OCR=false`, keep `OCR_PSM=6`, and limit `extract_max_files`. If that recovers values, adjust header detection/clamp rather than leaving OCR on.

- **UNHCR present but TBBC missing for a month**  
  Geometry should supplement TBBC via embedded text; if still missing, momentarily enable OCR for that window.

- **Two rows for the same camp (aliases)**  
  Update `data/reference/camp_lineage.csv` or `CAMP_ALIASES` (e.g., map “Nupo” → “Nu Po”), then re-run the same window. The extractor will unify names before writing CSVs.

- **WIDE looks flipped**  
  Confirm you’re looking at **WIDE** (not WIDE_DUAL). WIDE must have `report_date` as the first column.

- **Run got slower**  
  Reconfirm OCR is off, keep `extract_max_files` small, and use `skip_crawl=true` during extractor debugging.

---

## (Optional) Small YAML cleanups

- **`stop_after_found` mapping** — current line:
  ~~~yaml
  STOP_AFTER_FOUND: ${{ inputs.stop_after_found != '0' }}
  ~~~
  Prefer one of:
  ~~~yaml
  STOP_AFTER_FOUND: ${{ inputs.stop_after_found }}
  # or
  STOP_AFTER_FOUND: "${{ inputs.stop_after_found || 'true' }}"
  ~~~

- **`reliefweb_fallback`** — echoed but not defined/passed.  
  If you intend to use it, add an input:
  ~~~yaml
  reliefweb_fallback:
    description: "Use ReliefWeb API if TBC URL is missing"
    type: boolean
    required: false
    default: true
  ~~~
  and pass it to the Crawl step env:
  ~~~yaml
  RELIEFWEB_FALLBACK: "${{ inputs.reliefweb_fallback || 'true' }}"
  ~~~

---

## Link from README

Add this line anywhere in `README.md`:

```
See the [Action Quick Start](docs/action-quickstart.md) for run presets, input explanations, and troubleshooting.
```
