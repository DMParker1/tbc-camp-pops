# TBC Camp Populations

![scrape status](https://github.com/DMParker1/tbc-camp-pops/actions/workflows/scrape.yml/badge.svg)

_Last updated: 2025-08-14 03:13 UTC_

**Latest month:** **2022-07**

### Yearly table (camps as rows)
Values are the **latest available month** within each year (forward-filled inside the year).

| Camp            | 2011   | 2012   | 2013   | 2014   | 2015   | 2016   | 2017   | 2018   | 2019   | 2022   |
|:----------------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|
| Ban Don Yang    | 2,808  | 2,592  | 2,449  | 2,968  | 3,008  | 2,789  | 610    | 801    | 8,018  | 8,080  |
| Ban Mae Surin   |        | 1,860  |        |        | 2,564  |        | 2,282  | 15,336 | 14,264 | 14,162 |
| Ban Mai Nai Soi | 11,071 | 10,306 | 9,577  | 11,500 | 10,455 | 9,994  | 9,728  | 9,282  | 8,779  | 8,084  |
| Mae La          | 27,629 | 26,333 | 25,156 | 39,978 | 38,288 | 37,261 | 36,613 | 35,433 | 35,348 | 34,157 |
| Mae La Oon      | 10,136 | 9,449  | 8,675  | 10,517 | 10,283 | 9,855  | 9,546  | 9,464  | 9,212  | 8,940  |
| Mae Ra Ma Luang | 10,239 | 9,537  |        |        | 12,098 |        | 10,600 | 10,533 | 10,271 | 9,845  |
| Nu Po           | 8,914  | 8,544  | 7,927  | 11,253 | 11,320 | 11,064 | 10,646 | 10,452 | 10,093 | 7,524  |
| Tham Hin        | 4,281  | 4,268  | 4,314  | 6,300  | 6,480  | 6,223  | 6,184  | 6,182  | 6,147  | 5,734  |
| Umpiem Mai      | 11,017 | 10,975 |        |        | 12,658 |        |        | 11,265 | 11,129 | 10,657 |

## Data files

- `data/derived/tbc_camp_population_long.csv` — long/tidy (all months, all camps)
- `data/derived/tbc_camp_population_wide.csv` — pivot (camps as columns)

<!-- WORKFLOW_INPUTS_START -->
### Workflow inputs (quick reference)
- extract_since: YYYY-MM-DD (lower bound for extraction, inclusive)
- extract_until: YYYY-MM-DD (upper bound for extraction, inclusive)
- extract_max_files: integer (0 = skip extract; still build README)
- process_order: oldest|newest (applies to both crawler candidate months and extractor order)
- skip_crawl: true|false (true = reuse existing sources_index.csv)
- seed_only: true|false (crawler: only hit seed pages / search pages)
- stop_after_found: 0|1 (crawler: stop after first OK candidate per month; 0 = keep probing)
- scrape_since: YYYY-MM-DD (candidate generator lower bound; default 1992-01-01)
- scrape_until: YYYY-MM-DD (candidate generator upper bound; blank = today)
- reliefweb_fallback: true|false (use ReliefWeb API if TBC URL is missing)
- max_pages: integer (archive crawl page budget)
- print_every: integer (archive crawl progress cadence)

**Notes:**
- Preferred wide = TBC/TBBC over UNHCR when both exist for a month.
- Dual wide (`tbc_camp_population_wide_dual.csv`) retains both series for QA.
- Optional lineage file `data/reference/camp_lineage.csv` can narrow camp matching by active date ranges.
<!-- WORKFLOW_INPUTS_END -->
