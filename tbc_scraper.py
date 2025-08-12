#!/usr/bin/env python3
"""
Step 1: Crawl TBC Camp Population pages and index monthly map files (no OCR/extraction yet).
Outputs: data/derived/sources_index.csv
"""

import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import SSLError
import urllib3

# Silence warnings if we fall back to verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://www.theborderconsortium.org/"
START_URLS = [
    # Main index
    "https://www.theborderconsortium.org/resources/key-resources/camp-population/",
    # Yearly categories (wide net)
    *[f"https://www.theborderconsortium.org/category/camp-populations-{y}/" for y in range(1990, 2031)],
    # Site searches that often surface older media/posts
    "https://www.theborderconsortium.org/?s=camp+population+map",
    "https://www.theborderconsortium.org/?s=camp+populations",
    "https://www.theborderconsortium.org/?s=map",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TBC-Scraper-Index/1.1; +github.com/DMParker1/tbc-camp-pops)",
    "Accept-Language": "en-US,en;q=0.9",
}

# If you want to force strict TLS, set env TBC_VERIFY_SSL=true in the workflow step
VERIFY_DEFAULT = os.getenv("TBC_VERIFY_SSL", "false").lower() == "true"

# --- Updated month parser that handles numeric and month-name formats ---
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

def _century_from_two_digit(yy: int) -> int:
    """Map 90–99 → 1990s; 00–24 → 2000–2024; else 2000+yy (simple future-proof)."""
    if 90 <= yy <= 99:
        return 1900 + yy
    elif 0 <= yy <= 24:
        return 2000 + yy
    else:
        return 2000 + yy

def parse_report_date(s: str):
    """Return (YYYY-MM-01, method) or (None, None)."""
    if not s:
        return None, None
    s_clean = s.lower()

    # 1) YYYY[-_/.]MM
    m1 = re.search(r"(?P<y>(?:19|20)\d{2})[^0-9]?(?P<m>0[1-9]|1[0-2])", s_clean)
    if m1:
        y = int(m1.group("y")); mm = int(m1.group("m"))
        return f"{y:04d}-{mm:02d}-01", "yyyy-mm"

    # 2) Month name + 4-digit year OR year + month name
    m2 = re.search(
        r"(?:(?P<mname>[a-z]{3,9})[^0-9]{0,3}(?P<y4>(?:19|20)\d{2}))|"
        r"(?:(?P<y4b>(?:19|20)\d{2})[^a-z]{0,3}(?P<mnameb>[a-z]{3,9}))",
        s_clean
    )
    if m2:
        mname = (m2.group("mname") or m2.group("mnameb") or "").lower()
        y4 = m2.group("y4") or m2.group("y4b")
        if mname in MONTHS and y4:
            return f"{int(y4):04d}-{MONTHS[mname]:02d}-01", "monthname-yyyy"

    # 3) Month name + 2-digit year
    m3 = re.search(
        r"(?:(?P<mname2>[a-z]{3,9})[^0-9]{0,3}(?P<y2>\d{2}))|"
        r"(?:(?P<y2b>\d{2})[^a-z]{0,3}(?P<mname2b>[a-z]{3,9}))",
        s_clean
    )
    if m3:
        mname = (m3.group("mname2") or m3.group("mname2b") or "").lower()
        y2 = m3.group("y2") or m3.group("y2b")
        if mname in MONTHS and y2:
            y_full = _century_from_two_digit(int(y2))
            return f"{y_full:04d}-{MONTHS[mname]:02d}-01", "monthname-yy"

    return None, None

# File filter: likely monthly map artifacts
EXT_OK = (".pdf", ".jpg", ".jpeg", ".png")
KEYWORDS = ("map", "camp", "population", "unhcr")

def looks_like_map_file(href: str) -> bool:
    if not href:
        return False
    href_l = href.lower()
    if not href_l.endswith(EXT_OK):
        return False
    return any(k in href_l for k in KEYWORDS)

def get(url):
    """Fetch URL with normal TLS; on SSLError retry with verify=False (only for this host)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, verify=VERIFY_DEFAULT)
        r.raise_for_status()
        return r
    except SSLError as e:
        # Only retry insecurely for the TBC host
        if url.startswith(BASE):
            print(f"[warn] SSL verify failed for {url}; retrying without verification...", file=sys.stderr)
            r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
            r.raise_for_status()
            return r
        raise

def soup(url):
    try:
        return BeautifulSoup(get(url).text, "lxml")
    except Exception as e:
        print(f"[warn] fetch failed: {url} -> {e}", file=sys.stderr)
        return None

def crawl(max_pages=2000):
    to_visit = list(dict.fromkeys(START_URLS))  # de-dup while preserving order
    seen_pages = set()
    found = []  # dicts: report_date, source_url, file_name, origin_page, date_parse_method

    while to_visit and len(seen_pages) < max_pages:
        url = to_visit.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)

        s = soup(url)
        if not s:
            continue

        for a in s.select("a[href]"):
            href = a.get("href")
            if not href:
                continue

            # normalize to absolute for same-domain
            if href.startswith("/"):
                href = urljoin(BASE, href)

            # queue more pages on same domain
            if href.startswith(BASE) and href not in seen_pages and href not in to_visit:
                if any(seg in href for seg in ("/resources/", "/category/", "/media/", "/wp-content/", "/?s=")):
                    to_visit.append(href)

            # collect candidate files
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

        time.sleep(0.6)  # be polite

    # Build DataFrame with schema, even if no rows
    cols = ["report_date", "source_url", "file_name", "origin_page", "date_parse_method"]
    df = pd.DataFrame(found, columns=cols)

    if df.empty:
        return df  # zero rows but with headers

    # Dedup + sort
    df = df.sort_values(["source_url", df["report_date"].eq("").astype(int)]).drop_duplicates("source_url", keep="first")
    with pd.option_context('mode.use_inf_as_na', True):
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.sort_values(["report_date", "file_name"], na_position="last")
    df["report_date"] = df["report_date"].dt.strftime("%Y-%m-01").fillna("")
    return df

def main():
    out = Path("data/derived/sources_index.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = crawl()
    df.to_csv(out, index=False)
    print(f"Wrote {out.resolve()} with {len(df)} rows")

if __name__ == "__main__":
    main()
