# TBC Camp Populations

![scrape status](https://github.com/DMParker1/tbc-camp-pops/actions/workflows/scrape.yml/badge.svg)

_Last updated: 2025-08-13 16:24 UTC_

**Latest month:** **2025-04**

### Yearly table (camps as rows)
Values are the **latest available month** within each year (forward-filled inside the year).

| Camp            | 2011   | 2012   | 2013   | 2014   | 2015   | 2016   | 2017   | 2018   | 2019   | 2020   | 2021   | 2022   | 2023   | 2024   | 2025   |
|:----------------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|
| Ban Don Yang    | 808    | 2,592  | 2,449  | 2,437  | 3,008  | 2,789  | 2,747  |        |        |        |        |        | 2,437  | 2,437  | 2,437  |
| Ban Mae Surin   | 916    | 1,830  | 1,430  | 1,218  | 2,519  | 2,406  | 2,287  |        |        |        |        |        | 1,897  | 1,897  | 1,897  |
| Ban Mai Nai Soi |        |        |        |        |        |        | 9,730  |        |        |        |        |        | 9,799  | 9,799  | 9,799  |
| Mae La          | 10,136 | 49,606 | 28,675 |        | 10,283 | 10,031 | 36,613 | 35,666 | 34,718 | 34,320 | 34,215 | 34,063 | 34,063 | 34,063 | 34,063 |
| Mae La Oon      | 10,136 | 9,449  | 28,675 |        | 10,283 | 9,855  | 9,546  |        |        |        |        |        | 8,909  | 8,909  | 8,909  |
| Mae Ra Ma Luang | 10,239 | 9,268  |        |        |        |        | 10,592 |        |        |        |        |        | 9,799  | 9,799  | 9,799  |
| Tham Hin        | 4,279  | 4,268  | 4,314  | 4,335  | 6,480  | 6,223  | 6,168  |        |        |        |        |        | 5,712  | 5,712  | 5,712  |
| Umpiem Mai      | 11,017 | 10,443 | 39,890 |        |        |        | 11,586 | 11,296 |        |        | 4,464  |        | 10,609 | 10,609 | 10,609 |


## Workflow inputs (what each button does)

- **extract_since** (YYYY-MM-DD, optional)  
  Lower bound for documents by their report date. Example: `2019-01-01`.

- **extract_until** (YYYY-MM-DD, optional)  
  Upper bound for documents by their report date. Example: `2019-12-31`.  
  Leave blank to process through the latest.

- **process_order** (`oldest` | `newest`)  
  The order in which candidate files are processed within the selected date window.

- **extract_max_files** (integer)  
  Cap on how many new documents to process this run. Set `0` to do nothing (useful for README-only rebuilds).

- **skip_crawl** (`true` | `false`)  
  If `true`, reuse the existing `data/derived/sources_index.csv` and **don’t** hit the website.  
  Use this for backfills or quick tests.

- **seed_only** (`true` | `false`)  
  When crawling (if `skip_crawl=false`), only visit the seed pages (faster, limited coverage).  
  Has no effect when `skip_crawl=true`.

- **stop_after_found** (integer)  
  When crawling, stop after this many candidate links are found (0 = no limit).

- **RESUME_MODE** (env, fixed `true` in workflow)  
  Skip any (source_url, file_name) that have already been extracted into the long CSV.

- **OCR_DPI** (env, default `200`)  
  DPI used when rasterizing PDFs for OCR; higher is slower but can improve accuracy.

## Resulting Data files

- [data/derived/tbc_camp_population_long.csv](https://github.com/DMParker1/tbc-camp-pops/blob/main/data/derived/tbc_camp_population_long.csv) — long/tidy (all months, all camps)
- [data/derived/tbc_camp_population_wide.csv](https://github.com/DMParker1/tbc-camp-pops/blob/main/data/derived/tbc_camp_population_wide.csv) — pivot (camps as columns)

