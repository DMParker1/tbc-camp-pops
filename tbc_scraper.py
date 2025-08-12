#!/usr/bin/env python3
"""
Step 1: Crawl TBC Camp Population pages and index monthly map files (no OCR/extraction yet).
Writes: data/derived/sources_index.csv
"""

import os, re, sys, time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import SSLError
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://www.theborderconsortium.org/"
CUR_YEAR = datetime.utcnow().year

START_URLS = [
    "https://www.theborderconsortium.org/resources/key-resources/camp-population/",
    *[f"https://www.theborderconsortium.org/category/camp-populations-{y}/" for y in range(1990, CUR_YEAR + 1)],
    "https://www.theborderconsortium.org/?s=camp+population+map",
    "https://www.theborderconsortium.org/?s=camp+populations",
    "https://www.theborderconsortium.org/?s=map",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TBC-Scraper-Index/1.5; +github.com/DMParker1/tbc-camp-pops)",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_PAGES     = int(os.getenv("MAX_PAGES", "800"))
PRINT_EVERY   = int(os.getenv("PRINT_EVERY", "25"))
SEED_ONLY     = os.getenv("SEED_ONLY", "false").lower() == "true"
VERIFY_STRICT = os.getenv("TBC_VERIFY_SSL", "false").lower() == "true"

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

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
        if mname in MONTHS and y4: return f"{int(y4):04d}-{MONTHS[mname]:02d}-01", "monthname-yyyy"
    m3 = re.search(r"(?:(?P<mname2>[a-z]{3,9})[^0-9]{0,3}(?P<y2>\d{2}))|(?:(?P<y2b>\d{2})[^a-z]{0,3}(?P<mname2b>[a-z]{3,9}))", s_clean)
    if m3:
        mname = (m3.group("mname2") or m3.group("mname2b") or "").lower()
        y2 = m3.group("y2") or m3.group("y2b")
        if mname in MONTHS and y2:
            y_full = _century_from_two_digit(int(y2))
            return f"{y_full:04d}-{MONTHS[mname]:02d}-01", "monthname-yy"
    return None, None

EXT_OK = (".pdf", ".jpg", ".jpeg", ".png")

def looks_like_map_file(href: str) -> bool:
    if not href: return False
    href_l = href.lower()
    if not href_l.endswith(EXT_OK): return False
    # Heuristic: must contain "map" plus at least one of the other terms
    if "map" not in href_l: return False
    return any(k in href_l for k in ("camp", "population", "unhcr"))

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

def crawl(max_pages: int = MAX_PAGES) -> pd.DataFrame:
    to_visit = list(dict.fromkeys(START_URLS))
    seen_pages = set()
    found = []

    while to_visit and len(seen_pages) < max_pages:
        url = to_visit.pop(0)
        if url in seen_pages: continue
        seen_pages.add(url)

        if PRINT_EVERY and (len(seen_pages) % PRINT_EVERY == 0 or len(seen_pages) == 1):
            print(f"[progress] visited={len(seen_pages)} queued={len(to_visit)} found={len(found)} now={url}", flush=True)

        s = soup(url)
        if not s: continue

        for a in s.select("a[href]"):
            href = a.get("href") or ""
            if not href: continue
            if href.startswith("/"): href = urljoin(BASE, href)

            # enqueue ONLY likely HTML pages (avoid media)
            if (not SEED_ONLY
                and href.startswith(BASE)
                and href not in seen_pages and href not in to_visit):
                lower = href.lower()
                is_media = lower.endswith(EXT_OK)
                is_htmlish = any(seg in lower for seg in ("/resources/", "/category/", "/?s="))
                if is_htmlish and not is_media:
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

        time.sleep(0.25)

    cols = ["report_date", "source_url", "file_name", "origin_page", "date_parse_method"]
    df = pd.DataFrame(found, columns=cols)
    if df.empty: return df

    df["has_date"] = df["report_date"].astype(str).str.len().gt(0).astype(int)
    df = df.sort_values(["source_url", "has_date"], ascending=[True, False]).drop_duplicates("source_url", keep="first")
    df = df.drop(columns=["has_date"])

    with pd.option_context("mode.use_inf_as_na", True):
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.sort_values(["report_date", "file_name"], na_position="last")
    df["report_date"] = df["report_date"].dt.strftime("%Y-%m-01").fillna("")
    return df

def main():
    print(f"[start] running tbc_scraper.py (SEED_ONLY={SEED_ONLY} MAX_PAGES={MAX_PAGES} PRINT_EVERY={PRINT_EVERY} VERIFY_STRICT={VERIFY_STRICT})", flush=True)
    out = Path("data/derived/sources_index.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = crawl()
    df.to_csv(out, index=False)
    print(f"Wrote {out.resolve()} with {len(df)} rows", flush=True)

if __name__ == "__main__":
    main()
