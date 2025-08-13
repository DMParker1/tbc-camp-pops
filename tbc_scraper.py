#!/usr/bin/env python3
"""
TBC camp-populations — Scraper/Indexer (drop-in replacement)

What this version adds
- **Archive pagination** of the Camp Population pages (and year categories) to harvest linked PDFs/JPGs.
- **Per-month candidate generation** with HEAD/GET existence checks for older naming patterns (1998–2017) and newer uploads (2018+).
- **Very old scans**: probes `/wp-content/uploads/2021/03/Mon-YY-camp(s).pdf` for 1990s items.
- **ReliefWeb fallback (optional, default ON)** via public API to grab a month’s PDF when the TBC URL 404s/stalls.
- **Date bounds** via env: `SCRAPE_SINCE`, `SCRAPE_UNTIL` (YYYY-MM-DD). Order via `PROCESS_ORDER=oldest|newest`.
- **Resume-aware**: dedupes against existing `data/derived/sources_index.csv`.

Outputs
  data/derived/sources_index.csv  with columns:
    report_date, source_url, file_name, origin_page, date_parse_method

Env knobs (sensible defaults)
  MAX_PAGES=800         # archive crawl page budget
  PRINT_EVERY=25        # progress print cadence
  SKIP_CRAWL=false      # if true, only do candidate generation + fallback
  SEED_ONLY=false       # if true, enqueue only seed/archive HTML pages (no deep crawl)
  VERIFY_STRICT=false   # SSL verification against site (set true if you need strict)
  RELIEFWEB_FALLBACK=true  # use ReliefWeb API fallback when month not found on TBC
  SCRAPE_SINCE=1992-01-01   # lower bound (inclusive)
  SCRAPE_UNTIL=            # upper bound (inclusive); default = today
  PROCESS_ORDER=newest      # or oldest (affects candidate month iteration)
  STOP_AFTER_FOUND=true     # stop candidate probing after first OK per month
"""

import os, re, sys, time, json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import SSLError, RequestException
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://www.theborderconsortium.org"
EXT_OK = (".pdf", ".jpg", ".jpeg", ".png")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TBC-Scraper-Index/2.0; +github.com/DMParker1/tbc-camp-pops)",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---- env knobs ----
MAX_PAGES        = int(os.getenv("MAX_PAGES", "800"))
PRINT_EVERY      = int(os.getenv("PRINT_EVERY", "25"))
SKIP_CRAWL       = os.getenv("SKIP_CRAWL", "false").lower() == "true"
SEED_ONLY        = os.getenv("SEED_ONLY", "false").lower() == "true"
VERIFY_STRICT    = os.getenv("TBC_VERIFY_SSL", "false").lower() == "true"
RELIEFWEB_FALLBK = os.getenv("RELIEFWEB_FALLBACK", "true").lower() == "true"
PROCESS_ORDER    = os.getenv("PROCESS_ORDER", "newest").lower()
STOP_AFTER_FOUND = os.getenv("STOP_AFTER_FOUND", "true").lower() == "true"
SCRAPE_SINCE     = os.getenv("SCRAPE_SINCE", "1992-01-01")
SCRAPE_UNTIL     = os.getenv("SCRAPE_UNTIL", "").strip()

# ---- paths ----
DERIVED = Path("data/derived")
OUT_CSV = DERIVED / "sources_index.csv"

# ---- month helpers ----
MONTHS = [
    ("01","jan","January","Jan"),("02","feb","February","Feb"),("03","mar","March","Mar"),
    ("04","apr","April","Apr"),("05","may","May","May"),("06","jun","June","Jun"),
    ("07","jul","July","Jul"),("08","aug","August","Aug"),("09","sep","September","Sep"),
    ("10","oct","October","Oct"),("11","nov","November","Nov"),("12","dec","December","Dec"),
]
MNAME2NUM = {a:i+1 for i,(_,a,_,_) in enumerate(MONTHS)}

# ---- date parsing from strings ----

def _century_from_two_digit(yy: int) -> int:
    if 90 <= yy <= 99: return 1900 + yy
    elif 0 <= yy <= 24: return 2000 + yy
    else: return 2000 + yy

def parse_report_date(s: str):
    if not s: return None, None
    s_clean = s.lower()
    m1 = re.search(r"(?P<y>(?:19|20)\d{2})[^0-9]?(?P<m>0[1-9]|1[0-2])", s_clean)
    if m1: return f"{int(m1.group('y')):04d}-{int(m1.group('m')):02d}-01", "yyyy-mm"
    m2 = re.search(r"(?:(?P<mname>[a-z]{3,9})[^0-9]{0,3}(?P<y4>(?:19|20)\d{2}))|(?:(?P<y4b>(?:19|20)\d{2})[^a-z]{0,3}(?P<mnameb>[a-z]{3,9}))", s_clean)
    if m2:
        mname = (m2.group("mname") or m2.group("mnameb") or "").lower()
        y4 = m2.group("y4") or m2.group("y4b")
        if mname in MNAME2NUM and y4: return f"{int(y4):04d}-{MNAME2NUM[mname]:02d}-01", "monthname-yyyy"
    m3 = re.search(r"(?:(?P<mname2>[a-z]{3,9})[^0-9]{0,3}(?P<y2>\d{2}))|(?:(?P<y2b>\d{2})[^a-z]{0,3}(?P<mname2b>[a-z]{3,9}))", s_clean)
    if m3:
        mname = (m3.group("mname2") or m3.group("mname2b") or "").lower()
        y2 = m3.group("y2") or m3.group("y2b")
        if mname in MNAME2NUM and y2:
            y_full = _century_from_two_digit(int(y2))
            return f"{y_full:04d}-{MNAME2NUM[mname]:02d}-01", "monthname-yy"
    return None, None

# ---- HTTP helpers ----

def head_or_get_exists(url: str, timeout=15) -> bool:
    try:
        r = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True, verify=VERIFY_STRICT)
        if r.status_code == 200:
            return True
        if r.status_code in (403, 405):
            r2 = requests.get(url, headers=HEADERS, timeout=timeout, stream=True, verify=VERIFY_STRICT)
            ok = (r2.status_code == 200)
            try:
                r2.close()
            except Exception:
                pass
            return ok
    except SSLError:
        # retry without verification for TBC domain
        if url.startswith(BASE):
            try:
                r = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True, verify=False)
                if r.status_code == 200:
                    return True
                if r.status_code in (403, 405):
                    r2 = requests.get(url, headers=HEADERS, timeout=timeout, stream=True, verify=False)
                    ok = (r2.status_code == 200)
                    try: r2.close()
                    except Exception: pass
                    return ok
            except Exception:
                return False
    except RequestException:
        return False
    return False

# ---- Candidate URL generator ----

def candidate_urls(year: int, month: int) -> list:
    mm, mon, Month, Mon = MONTHS[month-1]
    y = f"{year:04d}"
    yy = f"{year%100:02d}"
    return [
        # media-era (c. 2012–2017)
        f"{BASE}/media/{y}-{mm}-{mon}-map-tbc-unhcr-1-.pdf",
        f"{BASE}/media/{y}-{mm}-{mon}-map-tbc-unhcr.pdf",
        # tbbc-era (c. 2008–2012)
        f"{BASE}/media/{y}-{mm}-{mon}-map-tbbc-unhcr-1-.pdf",
        f"{BASE}/media/{y}-{mm}-{mon}-map-tbbc-unhcr.pdf",
        # early media (c. 2001–2007)
        f"{BASE}/media/map-{y}-{mm}-{mon}-ccsdpt-tbbc-1-.pdf",
        f"{BASE}/media/map-{y}-{mm}-{mon}-1-.pdf",
        f"{BASE}/media/map-{y}-{mm}-{mon}.pdf",
        # uploads-era (2018+)
        f"{BASE}/wp-content/uploads/{y}/{mm}/{y}-{mm}-{Month}-map-tbc-unhcr.pdf",
        f"{BASE}/wp-content/uploads/{y}/{mm}/{y}-{mm}-{Month}-map-tbbc-unhcr.pdf",
        # very old scans (bulk uploaded Mar 2021)
        f"{BASE}/wp-content/uploads/2021/03/{Mon}-{yy}-camp.pdf",
        f"{BASE}/wp-content/uploads/2021/03/{Mon}-{yy}-camps.pdf",
    ]

# ---- ReliefWeb API fallback ----
API_URL = "https://api.reliefweb.int/v1/reports"

def reliefweb_find_pdf(y: int, m: int) -> Optional[str]:
    if not RELIEFWEB_FALLBK:
        return None
    from_d = f"{y:04d}-{m:02d}-01"
    # pick a liberal end-of-month; API supports YYYY-MM-DD windows
    to_d   = f"{y:04d}-{m:02d}-28"
    payload = {
        "appname": "tbc-camp-pops",
        "limit": 10,
        "profile": "full",
        "filter": {
            "operator": "AND",
            "conditions": [
                {"field": "source.name", "value": "The Border Consortium"},
                {"field": "format.name", "value": "Map"},
                {"field": "date.original", "value": {"from": from_d, "to": to_d}},
            ],
        },
        "sort": ["-date.original"],
    }
    try:
        r = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=20)
        r.raise_for_status()
        data = r.json()
        for item in data.get("data", []):
            files = (item.get("fields", {}).get("file", []) or [])
            for f in files:
                url = f.get("url") or ""
                if url.lower().endswith(".pdf"):
                    return url
    except RequestException:
        return None
    except Exception:
        return None
    return None

# ---- Crawl archive pages ----

def looks_like_map_file(href: str) -> bool:
    if not href: return False
    href_l = href.lower()
    if not href_l.endswith(EXT_OK): return False
    if "map" not in href_l: return False
    return any(k in href_l for k in ("camp", "population", "unhcr", "tbc", "tbbc", "bbc"))

START_URLS_BASE = [
    f"{BASE}/resources/key-resources/camp-population/",
    # year-category pages (not all exist, but cheap to try)
    *[f"{BASE}/category/camp-populations-{y}/" for y in range(1990, datetime.utcnow().year + 1)],
]

SEARCH_QUERIES = [
    f"{BASE}/?s=camp+population+map",
    f"{BASE}/?s=camp+populations",
    f"{BASE}/?s=map",
]


def get(url: str) -> requests.Response:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, verify=VERIFY_STRICT)
        r.raise_for_status()
        return r
    except SSLError:
        if url.startswith(BASE):
            print(f"[warn] SSL verify failed for {url}; retrying without verification...", file=sys.stderr, flush=True)
            r = requests.get(url, headers=HEADERS, timeout=20, verify=False)
            r.raise_for_status()
            return r
        raise


def soup(url: str):
    try:
        html = get(url).text
        return BeautifulSoup(html, "lxml")
    except Exception as e:
        print(f"[warn] fetch failed: {url} -> {e}", file=sys.stderr, flush=True)
        return None


def crawl_archive(max_pages: int = MAX_PAGES) -> pd.DataFrame:
    if SKIP_CRAWL:
        return pd.DataFrame(columns=["report_date","source_url","file_name","origin_page","date_parse_method"])  # empty; we will generate

    to_visit = list(dict.fromkeys(START_URLS_BASE + SEARCH_QUERIES))
    seen_pages = set()
    found = []

    while to_visit and len(seen_pages) < max_pages:
        url = to_visit.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)

        if PRINT_EVERY and (len(seen_pages) % PRINT_EVERY == 0 or len(seen_pages) == 1):
            print(f"[progress] visited={len(seen_pages)} queued={len(to_visit)} found={len(found)} now={url}", flush=True)

        s = soup(url)
        if not s:
            continue

        for a in s.select("a[href]"):
            href = a.get("href") or ""
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(BASE, href)

            # enqueue ONLY likely HTML pages (avoid media)
            if (not SEED_ONLY and href.startswith(BASE) and href not in seen_pages and href not in to_visit):
                lower = href.lower()
                is_media = lower.endswith(EXT_OK)
                is_htmlish = any(seg in lower for seg in ("/resources/", "/category/", "/?s="))
                if is_htmlish and not is_media:
                    to_visit.append(href)

            if looks_like_map_file(href):
                fn = urlparse(href).path.split("/")[-1]
                date_str, method = parse_report_date(fn)
                if not date_str:
                    date_str, method = parse_report_date(href)
                found.append({
                    "report_date": date_str or "",
                    "source_url": href,
                    "file_name": fn,
                    "origin_page": url,
                    "date_parse_method": method or ""
                })

        time.sleep(0.2)

    cols = ["report_date","source_url","file_name","origin_page","date_parse_method"]
    df = pd.DataFrame(found, columns=cols)
    if df.empty:
        return df

    df["has_date"] = df["report_date"].astype(str).str.len().gt(0).astype(int)
    df = df.sort_values(["source_url","has_date"], ascending=[True, False]).drop_duplicates("source_url", keep="first")
    df = df.drop(columns=["has_date"])

    with pd.option_context("mode.use_inf_as_na", True):
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.sort_values(["report_date", "file_name"], na_position="last")
    df["report_date"] = df["report_date"].dt.strftime("%Y-%m-01").fillna("")
    return df

# ---- Candidate generator across a date window ----

def iter_months(since: str, until: str):
    s = datetime.strptime(since, "%Y-%m-%d").date()
    u = datetime.strptime(until, "%Y-%m-%d").date() if until else date.today()
    # normalize to first of month
    s = s.replace(day=1)
    u = u.replace(day=1)
    cur = u if PROCESS_ORDER != "oldest" else s
    step = -1 if PROCESS_ORDER != "oldest" else 1
    while True:
        yield cur.year, cur.month
        if (PROCESS_ORDER != "oldest" and (cur.year == s.year and cur.month == s.month)):
            break
        if (PROCESS_ORDER == "oldest" and (cur.year == u.year and cur.month == u.month)):
            break
        # increment
        if PROCESS_ORDER == "oldest":
            ny, nm = (cur.year + (1 if cur.month == 12 else 0)), (1 if cur.month == 12 else cur.month + 1)
        else:
            ny, nm = (cur.year - (1 if cur.month == 1 else 0)), (12 if cur.month == 1 else cur.month - 1)
        cur = date(ny, nm, 1)


def generate_candidates_df() -> pd.DataFrame:
    rows = []
    seen = set()
    for y, m in iter_months(SCRAPE_SINCE, SCRAPE_UNTIL or date.today().strftime("%Y-%m-%d")):
        urls = candidate_urls(y, m)
        picked = None
        for u in urls:
            if head_or_get_exists(u):
            picked = u
            if STOP_AFTER_FOUND:
                break
        # Optionally try ReliefWeb for the month
        if not picked:
            picked = reliefweb_find_pdf(y, m)
        if picked:
            if picked in seen:
                continue
            fn = urlparse(picked).path.split("/")[-1]
            rows.append({
                "report_date": f"{y:04d}-{m:02d}-01",
                "source_url": picked,
                "file_name": fn,
                "origin_page": "candidate-generator" if picked.startswith(BASE) else "reliefweb-fallback",
                "date_parse_method": "generator"
            })
            seen.add(picked)
    return pd.DataFrame(rows, columns=["report_date","source_url","file_name","origin_page","date_parse_method"])

# ---- Combine, resume, write ----

def merge_and_write(df_crawl: pd.DataFrame, df_gen: pd.DataFrame):
    DERIVED.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
        try:
            existing = pd.read_csv(OUT_CSV, dtype=str).fillna("")
        except Exception:
            existing = pd.DataFrame()

    df_all = pd.concat([existing, df_crawl, df_gen], ignore_index=True)
    if df_all.empty:
        df_all.to_csv(OUT_CSV, index=False)
        print(f"Wrote {OUT_CSV.resolve()} with 0 rows", flush=True)
        return

    # Deduplicate by URL, keep the best-dated row
    with pd.option_context("mode.use_inf_as_na", True):
        df_all["report_date"] = pd.to_datetime(df_all["report_date"], errors="coerce")
    df_all["has_date"] = df_all["report_date"].notna().astype(int)
    df_all = (df_all
              .sort_values(["source_url","has_date","report_date"], ascending=[True, False, True])
              .drop_duplicates("source_url", keep="last")
              .drop(columns=["has_date"]))
    df_all["report_date"] = df_all["report_date"].dt.strftime("%Y-%m-01").fillna("")

    df_all = df_all.sort_values(["report_date","file_name"], na_position="last")
    df_all.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV.resolve()} with {len(df_all)} rows", flush=True)

# ---- main ----

def main():
    print(f"[start] tbc_scraper.py SKIP_CRAWL={SKIP_CRAWL} PROCESS_ORDER={PROCESS_ORDER} SINCE={SCRAPE_SINCE} UNTIL={SCRAPE_UNTIL or 'today'} RELIEFWEB={RELIEFWEB_FALLBK}", flush=True)

    df_crawl = crawl_archive(MAX_PAGES)
    df_gen   = generate_candidates_df()

    merge_and_write(df_crawl, df_gen)

if __name__ == "__main__":
    main()
