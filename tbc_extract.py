#!/usr/bin/env python3
"""
Step 2: Download each source in data/derived/sources_index.csv and extract per-camp populations.

Key improvements:
- High-confidence parsing: capture numbers only within a short window AFTER each known camp label.
- Strict noise filters: ignore years (e.g., 2025), month words ("June"), totals, and weird OCR junk.
- Plausible ranges: global (50..300_000) and per-camp ranges (e.g., Mae La ~10k..70k).
- Resume + cache: skips already-processed files; reuses data/raw downloads.
- Wide CSV built with nullable ints (avoids overflow errors).

Outputs:
  - data/derived/tbc_camp_population_long.csv
  - data/derived/tbc_camp_population_wide.csv
  - data/raw/<files>  (cached artifacts)

Env knobs (set in workflow):
  - TBC_VERIFY_SSL: "true"|"false"   (default false; strict TLS if true)
  - EXTRACT_SINCE: "YYYY-MM-DD"      (filter index to this date or newer; blank = all)
  - EXTRACT_MAX_FILES: int           (max source files to process this run; default 250)
  - PROCESS_ORDER: "newest"|"oldest" (which end to process first; default newest)
  - RESUME_MODE: "true"|"false"      (skip files already present in LONG CSV; default true)
  - OCR_DPI: int                     (DPI for PDF->image OCR; default 200)
"""

import os, io, re, sys, time, logging
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import pandas as pd
import requests
import urllib3

# Verbose but suppress TLS chatter; quiet pdfminer noise
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# --- Config & env ---
RAW_DIR      = Path("data/raw")
DERIVED_DIR  = Path("data/derived")
INDEX_CSV    = DERIVED_DIR / "sources_index.csv"
OUT_LONG     = DERIVED_DIR / "tbc_camp_population_long.csv"
OUT_WIDE     = DERIVED_DIR / "tbc_camp_population_wide.csv"

EXTRACT_SINCE     = os.getenv("EXTRACT_SINCE", "").strip()
EXTRACT_MAX_FILES = int(os.getenv("EXTRACT_MAX_FILES", "250"))
PROCESS_ORDER     = os.getenv("PROCESS_ORDER", "newest").lower()
RESUME_MODE       = os.getenv("RESUME_MODE", "true").lower() == "true"
OCR_DPI           = int(os.getenv("OCR_DPI", "200"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TBC-Extractor/1.3; +github.com/DMParker1/tbc-camp-pops)",
    "Accept-Language": "en-US,en;q=0.9",
}

# Known camps (canonical labels)
KNOWN_CAMPS = [
    "Ban Mai Nai Soi","Ban Mae Surin","Mae La Oon","Mae Ra Ma Luang",
    "Mae La","Umpiem Mai","Nupo","Ban Don Yang","Tham Hin"
]

# Common OCR alias forms -> canonical
ALIASES = {
    "ban mainai soi": "Ban Mai Nai Soi",
    "ban mai nai soi": "Ban Mai Nai Soi",
    "ban mae sur in": "Ban Mae Surin",
    "mae lao on": "Mae La Oon",
    "mae la oon": "Mae La Oon",
    "mae ra ma luang": "Mae Ra Ma Luang",
    "mae ra ma luan g": "Mae Ra Ma Luang",
    "mae la": "Mae La",
    "umpiem mai": "Umpiem Mai",
    "um pie m": "Umpiem Mai",
    "nupo": "Nupo",
    "ban don yang": "Ban Don Yang",
    "tham hin": "Tham Hin",
    "thamhin": "Tham Hin",
}

# Month words & stopwords to exclude as "labels"
MONTH_WORDS = {
    "jan","january","feb","february","mar","march","apr","april","may",
    "jun","june","jul","july","aug","august","sep","sept","september",
    "oct","october","nov","november","dec","december"
}
LABEL_STOPWORDS = {
    "map","unhcr","update","tbc","tbbc","total","grand total","province",
    "thai myanmar border","myanmar thailand border","thailand myanmar border"
}

# Global plausible population range (helps reject OCR junk)
GLOBAL_MIN = 50
GLOBAL_MAX = 300_000

# Per-camp plausible ranges (broad on purpose; tune if needed)
CAMP_BOUNDS = {
    "Mae La":          (10_000,  70_000),
    "Umpiem Mai":      (3_000,   45_000),
    "Nupo":            (2_000,   40_000),
    "Mae Ra Ma Luang": (2_000,   50_000),
    "Mae La Oon":      (2_000,   40_000),
    "Ban Mai Nai Soi": (1_000,   30_000),
    "Ban Mae Surin":   (500,     20_000),
    "Ban Don Yang":    (500,     15_000),
    "Tham Hin":        (1_000,   25_000),
}

# --- HTTP (env-aware SSL + fallback) ---
def get(url):
    verify_pref = os.getenv("TBC_VERIFY_SSL", "false").lower() == "true"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, verify=verify_pref)
        r.raise_for_status()
        return r
    except requests.exceptions.SSLError:
        if url.startswith("https://www.theborderconsortium.org/"):
            print(f"[warn] SSL verify failed for {url}; retrying without verification...", flush=True)
            r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
            r.raise_for_status()
            return r
        raise

def safe_filename(url, report_date):
    name = urlparse(url).path.split("/")[-1] or "file"
    if report_date:
        name = f"{report_date}_{name}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def download(url, report_date):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / safe_filename(url, report_date)
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        data = get(url).content
    except requests.exceptions.HTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 404:
            print(f"[info] 404 not found, skipping: {url}", flush=True)
            return None
        raise
    out.write_bytes(data)
    return out

# --- Text extraction ---
def extract_text_from_pdf(pdf_path):
    txt_all = []
    method = "pdf-text"
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    txt_all.append(t)
        combined = "\n".join(txt_all)
        # If text is too sparse, OCR the images
        if len(combined.strip()) >= 300:
            return combined, method
        else:
            print(f"[info] {pdf_path.name}: pdf text sparse; falling back to OCR", flush=True)
    except Exception as e:
        print(f"[warn] pdfplumber failed on {pdf_path.name}: {e}", flush=True)

    try:
        import fitz  # PyMuPDF
        from PIL import Image, ImageOps, ImageFilter
        import pytesseract
        method = "pdf-ocr"
        doc = fitz.open(pdf_path)
        parts = []
        for p in doc:
            pix = p.get_pixmap(dpi=OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
            img = ImageOps.autocontrast(img)
            img = img.filter(ImageFilter.SHARPEN)
            parts.append(pytesseract.image_to_string(img))
        return "\n".join(parts), method
    except Exception as e:
        print(f"[warn] pdf OCR failed on {pdf_path.name}: {e}", flush=True)
        return "", "pdf-fail"

def extract_text_from_image(img_path):
    try:
        from PIL import Image, ImageOps, ImageFilter
        import pytesseract
        img = Image.open(img_path).convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
        txt = pytesseract.image_to_string(img)
        return txt, "img-ocr"
    except Exception as e:
        print(f"[warn] image OCR failed on {img_path.name}: {e}", flush=True)
        return "", "img-fail"

# --- Helpers ---
def normalize(s):
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[^\S\r\n]+", " ", s)
    return s

def detect_category(text):
    if re.search(r"\bIDP\b|Internally Displaced", text, re.I):
        return "idp"
    return "refugee"

def canon_label(raw_label: str):
    """Normalize and map OCR label -> canonical camp name when possible."""
    if not raw_label:
        return None
    norm = re.sub(r"[^a-z]", " ", raw_label.lower()).strip()
    key = re.sub(r"\s+", " ", norm)
    if key in ALIASES:
        return ALIASES[key]
    # direct match after normalization
    for camp in KNOWN_CAMPS:
        if re.sub(r"[^a-z]", "", camp.lower()) == re.sub(r"[^a-z]", "", raw_label.lower()):
            return camp
    return None

def build_camp_patterns():
    """Compile regex patterns that match each camp label with minor OCR spacing/hyphen noise."""
    pats = {}
    for camp in KNOWN_CAMPS:
        esc = re.escape(camp)
        # Allow spaces/hyphens/underscores variability between words
        tokenized = r"[ _\-]*".join([re.escape(t) for t in camp.split()])
        pats[camp] = re.compile(rf"(?i)\b{tokenized}\b")
    return pats

CAMP_PATS = build_camp_patterns()

def extract_numbers_in_window(text_slice: str):
    """
    Return list of plausible ints found in a small window of text.
    - Accept thousand separators , . or spaces; strip to digits.
    - Reject years (1900..2200) and out-of-global-range values.
    """
    nums = []
    for m in re.finditer(r"([0-9][0-9,\.\s]{1,12})", text_slice):
        digits = re.sub(r"[^\d]", "", m.group(1))
        if not digits or not digits.isdigit():
            continue
        val = int(digits)
        # reject obvious years and out-of-range junk
        if 1900 <= val <= 2200:
            continue
        if not (GLOBAL_MIN <= val <= GLOBAL_MAX):
            continue
        nums.append(val)
    return nums

def is_household_hint(around: str) -> bool:
    """Return True if nearby text suggests household counts, not population."""
    return bool(re.search(r"\bHH\b|\bhouseholds?\b|\bhh:?\b", around, re.I))

# --- Parsing core ---
def parse_rows(text):
    """
    Extract rows as dicts:
      camp_name, population, category, parse_notes, parse_confidence
    Strategy:
      - For each known camp, search label; look +N chars ahead for plausible numbers.
      - Prefer the largest plausible number near the label (often population vs HH).
      - Enforce per-camp plausible bounds; drop if outside.
      - As a fallback, consider generic label-number pairs but with strict filters.
    """
    rows = []
    clean = normalize(text)
    category = detect_category(clean)

    # 1) High-confidence pass: known camps only
    for camp, pat in CAMP_PATS.items():
        for m in pat.finditer(clean):
            start = m.end()
            window = clean[start:start+96]  # small forward window after label
            if not window.strip():
                continue
            # Drop if this window screams "total"
            if re.search(r"\b(total|grand total)\b", window, re.I):
                continue
            # Collect plausible numbers
            candidates = extract_numbers_in_window(window)
            if not candidates:
                continue
            # Avoid household counts if we can detect them
            if is_household_hint(window):
                # If HH hinted, maybe next window contains pop; peek a bit further
                window2 = clean[start:start+160]
                c2 = extract_numbers_in_window(window2)
                if c2:
                    candidates = c2
            # Pick the largest plausible value near the camp (often the population)
            val = max(candidates) if candidates else None
            if val is None:
                continue

            # Per-camp sanity check
            mn, mx = CAMP_BOUNDS.get(camp, (GLOBAL_MIN, GLOBAL_MAX))
            if not (mn <= val <= mx):
                # Value outside camp bounds — drop as likely noise
                continue

            rows.append({
                "camp_name": camp,
                "population": int(val),
                "category": category,
                "parse_notes": "camp_window_peak",
                "parse_confidence": 0.95
            })
            break  # stop after first valid capture for this camp

    # 2) Fallback pass (generic "Label 12345"), very strict to avoid noise
    #    Only run if we found nothing at all (e.g., IDP maps or unknown labels)
    if False and not rows:
        seen = set()
        for m in re.finditer(r"([A-Z][A-Za-z /'\-]{2,40})\s+([0-9][0-9,\.\s]{2,})", clean):
            raw_label = m.group(1).strip()
            digits = re.sub(r"[^\d]", "", m.group(2))
            if not digits.isdigit():
                continue
            val = int(digits)

            # reject obvious non-population numbers
            if 1900 <= val <= 2200:
                continue
            if not (GLOBAL_MIN <= val <= GLOBAL_MAX):
                continue

            # normalize label; drop month words & stopwords
            norm = re.sub(r"[^a-z]", " ", raw_label.lower()).strip()
            key = re.sub(r"\s+", " ", norm)
            if key in MONTH_WORDS or key in LABEL_STOPWORDS:
                continue

            # map to canonical camp if alias known; otherwise keep raw
            label = ALIASES.get(key, raw_label)
            if label in seen:
                continue
            seen.add(label)

            # If this accidentally matches a known camp, also check camp bounds
            if label in CAMP_BOUNDS:
                mn, mx = CAMP_BOUNDS[label]
                if not (mn <= val <= mx):
                    continue

            rows.append({
                "camp_name": label,
                "population": int(val),
                "category": category,
                "parse_notes": "generic_label_number",
                "parse_confidence": 0.6 if label in CAMP_BOUNDS else 0.5
            })

    # Deduplicate by camp_name within this document (keep highest population seen)
    if rows:
        best = {}
        for r in rows:
            name = r["camp_name"]
            if name not in best or (r.get("population") or 0) > (best[name].get("population") or 0):
                best[name] = r
        rows = list(best.values())

    return rows

# --- Main pipeline ---
def main():
    if not INDEX_CSV.exists() or INDEX_CSV.stat().st_size == 0:
        print(f"[error] missing or empty {INDEX_CSV}", flush=True)
        sys.exit(0)

    df_idx = pd.read_csv(INDEX_CSV, dtype=str).fillna("")
    if df_idx.empty:
        print("[warn] sources_index.csv has 0 rows", flush=True)
        pd.DataFrame(columns=[
            "report_date","camp_name","population","category",
            "source_url","file_name","extract_method","parse_notes","parse_confidence"
        ]).to_csv(OUT_LONG, index=False)
        pd.DataFrame().to_csv(OUT_WIDE, index=False)
        return

    # Parse/normalize dates
    with pd.option_context("mode.use_inf_as_na", True):
        df_idx["report_date"] = pd.to_datetime(df_idx["report_date"], errors="coerce")

    if EXTRACT_SINCE:
        try:
            since = pd.to_datetime(EXTRACT_SINCE)
            df_idx = df_idx[df_idx["report_date"] >= since]
        except Exception as e:
            print(f"[warn] bad EXTRACT_SINCE={EXTRACT_SINCE}: {e}", flush=True)

    # Resume: skip files we've already extracted
    existing = pd.DataFrame()
    processed_keys = set()
    if RESUME_MODE and OUT_LONG.exists() and OUT_LONG.stat().st_size:
        try:
            existing = pd.read_csv(OUT_LONG, dtype=str).fillna("")
            if not existing.empty:
                processed_keys = set(zip(existing.get("source_url", []), existing.get("file_name", [])))
                with pd.option_context("mode.use_inf_as_na", True):
                    existing["report_date"] = pd.to_datetime(existing["report_date"], errors="coerce")
        except Exception as e:
            print(f"[warn] could not read existing LONG CSV: {e}", flush=True)

    # Order
    df_idx = df_idx.dropna(subset=["source_url"])
    if PROCESS_ORDER == "oldest":
        df_idx = df_idx.sort_values(["report_date", "file_name"], na_position="last")
    else:
        df_idx = df_idx.sort_values(["report_date", "file_name"], na_position="last", ascending=[False, True])

    # Limit per run
    candidates = []
    for _, r in df_idx.iterrows():
        key = (str(r.get("source_url","")), str(r.get("file_name","")))
        if RESUME_MODE and key in processed_keys:
            continue
        candidates.append(r)
        if len(candidates) >= EXTRACT_MAX_FILES:
            break

    print(f"[info] extracting from {len(candidates)} new files "
          f"(resume={RESUME_MODE}, since='{EXTRACT_SINCE or 'ALL'}', order={PROCESS_ORDER})",
          flush=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    new_records = []

    for r in candidates:
        report_date = r["report_date"]
        url         = (r.get("source_url") or "").strip()
        fname       = (r.get("file_name")  or "").strip()
        rd_str      = report_date.strftime("%Y-%m-01") if pd.notna(report_date) else ""

        if not url:
            continue

        local = download(url, rd_str)
        if local is None:
            continue

        ext = local.suffix.lower()
        text, method = "", None
        try:
            if ext == ".pdf":
                text, method = extract_text_from_pdf(local)
            elif ext in (".jpg",".jpeg",".png"):
                text, method = extract_text_from_image(local)
            else:
                text, method = extract_text_from_image(local)
        except Exception as e:
            print(f"[warn] extraction failed: {local.name} -> {e}", flush=True)

        if not (text or "").strip():
            new_records.append({
                "report_date": rd_str, "camp_name": None, "population": None, "category": None,
                "source_url": url, "file_name": local.name, "extract_method": method or "none",
                "parse_notes": "no_text_extracted", "parse_confidence": 0.0
            })
            continue

        rows = parse_rows(text)
        if not rows:
            rows = [{
                "camp_name": None, "population": None, "category": None,
                "parse_notes": "no_rows_parsed", "parse_confidence": 0.0
            }]

        for row in rows:
            row.update({
                "report_date": rd_str,
                "source_url": url,
                "file_name": local.name,
                "extract_method": method or "unknown",
            })
            new_records.append(row)

        time.sleep(0.1)  # be polite

    # Combine with existing & write
    df_new = pd.DataFrame.from_records(new_records)
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new

    if not combined.empty:
        with pd.option_context("mode.use_inf_as_na", True):
            combined["report_date"] = pd.to_datetime(combined["report_date"], errors="coerce")
        # Drop exact duplicates
        combined = (combined
                    .drop_duplicates(subset=["report_date","camp_name","source_url","file_name","extract_method","parse_notes"],
                                     keep="last")
                    .sort_values(["report_date","camp_name"], na_position="last"))
        combined["report_date"] = combined["report_date"].dt.strftime("%Y-%m-01").fillna("")

    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_LONG, index=False)

    # Wide pivot (robust numeric handling)
    try:
        if not combined.empty:
            tmp = combined.dropna(subset=["camp_name","population"]).copy()
            tmp["population"] = pd.to_numeric(tmp["population"], errors="coerce")
            # Keep only plausible values
            tmp = tmp[tmp["population"].between(GLOBAL_MIN, GLOBAL_MAX)]
            # If camp has bounds, enforce them
            def within_bounds(row):
                camp = str(row["camp_name"])
                val  = row["population"]
                if camp in CAMP_BOUNDS and pd.notna(val):
                    lo, hi = CAMP_BOUNDS[camp]
                    return lo <= val <= hi
                return True
            tmp = tmp[tmp.apply(within_bounds, axis=1)]
            tmp["population"] = tmp["population"].astype("Int64")

            wide = (tmp
                    .sort_values(["report_date"])
                    .drop_duplicates(["report_date","camp_name"], keep="last")
                    .pivot(index="report_date", columns="camp_name", values="population")
                    .sort_index())
            wide.to_csv(OUT_WIDE)
        else:
            pd.DataFrame().to_csv(OUT_WIDE, index=False)
    except Exception as e:
        print(f"[warn] could not generate wide CSV: {e}", flush=True)
        pd.DataFrame().to_csv(OUT_WIDE, index=False)

    print(f"[done] wrote {OUT_LONG} (+{len(df_new)} new rows, total {len(combined)})", flush=True)

if __name__ == "__main__":
    main()
