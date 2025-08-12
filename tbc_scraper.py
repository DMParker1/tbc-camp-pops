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
