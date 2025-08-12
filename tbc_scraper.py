#!/usr/bin/env python3
"""
Step 1: Crawl TBC Camp Population pages and index monthly map files (no OCR/extraction yet).
Outputs: data/derived/sources_index.csv
"""

import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.theborderconsortium.org/"
START_URLS = [
    "https://www.theborderconsortium.org/resources/key-resources/camp-population/",
    *[f"https://www.theborderconsortium.org/category/camp-populations-{y}/" for y in range(1990, 2031)],
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TBC-Scraper-Index/1.0)"}

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

def _century_from_two_digit(yy: int) -> int:
    if 90 <= yy <= 99:
        return 1900 + yy
    elif 0 <= yy <= 24:
        return 2000 + yy
    else:
        return 2000 + yy

def parse_report_date(s: str):
    if not s:
        return None, None
    s_clean = s.lower()

    m1 = re.search(r"(?P<y>(?:19|20)\d{2})[^0-9]?(?P<m>0[1-9]|1[0-2])", s_clean)
    if m1:
        y = int(m1.group("y"))
        mm = int(m1.group("m"))
        return f"{y:04d}-{mm:02d}-01", "yyyy-mm"

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
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r

def soup(url):
    try:
        return BeautifulSoup(get(url).text, "lxml")
    except Exception as e:
        print(f"[warn] fetch failed: {url} -> {e}", file=sys.stderr)
        return None

def crawl(max_pages=800):
    to_visit = list(START_URLS)
    seen_pages = set()
    found = []

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

            if href.startswith("/"):
                href = urljoin(BASE, href)

            if href.startswith(BASE) and href not in seen_pages and href not in to_visit:
                if any(seg in href for seg in ("/resources/", "/category/", "/media/", "/wp-content/")):
                    to_visit.append(href)

            if looks_like_map_file(href):
                fn = urlparse(href).path.split("/")[-1]
                date_str, method = parse_report_date(fn) or (None, None)
                if not date_str:
                    date_str, method = parse_report_date(href) or (None, None)
                found.append({
                    "report_date": date_str or "",
                    "source_url": href,
                    "file_name": fn,
                    "origin_page": url,
                    "date_parse_method": method or ""
                })

        time.sleep(0.4)

    # Build DataFrame with schema, even if no rows
    schema = ["report_date", "source_url", "file_name", "origin_page", "date_parse_method"]
    df = pd.DataFrame(found, columns=schema)

    if df.empty:
        return df  # zero rows, but columns exist

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
