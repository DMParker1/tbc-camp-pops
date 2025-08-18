# TBC Camp Populations

![scrape status](https://github.com/DMParker1/tbc-camp-pops/actions/workflows/scrape.yml/badge.svg)

_Last updated: 2025-08-18 21:28 UTC_

**Latest month:** **2008-12**

### Yearly table (camps as rows)
Values are the **latest available month** within each year (forward-filled inside the year).

| Camp            | 1998   | 1999   | 2000   | 2001   | 2002   | 2003   | 2004   | 2005   | 2006   | 2007   | 2008   |
|:----------------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|
| Ban Don Yang    | 1,736  | 2,011  | 3,653  | 3,907  | 3,813  | 3,672  |        |        | 4,552  | 3,659  | 3,674  |
| Ban Mae Surin   | 2,772  | 2,861  | 2,959  | 3,240  | 3,521  | 3,460  | 1,850  | 3,494  | 111    | 3,448  | 3,531  |
| Ban Mai Nai Soi | 3,231  | 3,508  | 3,871  | 4,321  | 4,410  | 565    |        |        |        |        |        |
| Mae La          | 31,680 | 32,875 | 37,070 | 40,503 | 40,505 |        | 22,972 | 482    | 101    | 39,239 | 33,962 |
| Mae La Oon      |        |        |        |        |        |        |        | 15,015 |        |        |        |
| Mae Ra Ma Luang | 7,256  | 7,802  | 8,743  | 9,830  | 10,975 | 11,626 | 5,781  | 1,191  | 392    |        |        |
| Nu Po           | 8,817  | 8,107  | 8,777  | 9,612  | 11,596 | 12,345 | 6,460  |        |        |        | 11,515 |
| Tham Hin        | 8,306  | 7,748  | 8,526  | 9,149  | 8,867  |        |        |        |        |        | 5,089  |
| Umpiem Mai      | 2,655  | 16,300 | 16,085 | 16,758 | 17,336 |        |        | 131    | 197    | 19,851 | 14,505 |

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
