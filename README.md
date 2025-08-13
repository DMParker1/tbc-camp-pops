# TBC Camp Populations

![scrape status](https://github.com/DMParker1/tbc-camp-pops/actions/workflows/scrape.yml/badge.svg)

_Last updated: 2025-08-13 23:14 UTC_

**Latest month:** **2011-12**

### Yearly table (camps as rows)
Values are the **latest available month** within each year (forward-filled inside the year).

| Camp            | 2011   |
|:----------------|:-------|
| Ban Don Yang    | 2,808  |
| Ban Mai Nai Soi | 11,071 |
| Mae La          | 27,629 |
| Mae La Oon      | 10,136 |
| Mae Ra Ma Luang | 10,239 |
| Nu Po           | 8,914  |
| Tham Hin        | 4,281  |
| Umpiem Mai      | 11,017 |

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
