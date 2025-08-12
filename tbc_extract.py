#!/usr/bin/env python3
"""
Incremental extractor: downloads sources from data/derived/sources_index.csv,
parses per-camp populations, and appends only NEW results.

Outputs:
  - data/derived/tbc_camp_population_long.csv
  - data/derived/tbc_camp_population_wide.csv
  - data/raw/  (cached across runs via actions/cache)

Env knobs:
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

# Verbose but not noisy about TLS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# --- Config & env ---
RAW_DIR      = Path("data/raw")
DERIVED_DIR  = Path("data/derived")
INDEX_CSV    = DERIVED_DIR / "sources_index.csv"
OUT_LONG     = DERIVED_DIR / "tbc_camp_population_long.csv"
OUT_WIDE     = DERIVED_DIR / "tbc_camp_population_wide.csv"

EXTRACT_SINCE    = os.getenv("EXTRACT_SINCE", "").strip()
EXTRACT_MAX_FILES= int(os.getenv("EXTRACT_MAX_FILES", "250"))
PROCESS_ORDER    = os.getenv("PROCESS_ORDER", "newest").lower()
RESUME_MODE      = os.getenv("RESUME_MODE", "true").lower() == "true"
OCR_DPI          = int(os.getenv("OCR_DPI", "200"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TBC-Extractor/1.2; +github.com/DMParker1/tbc-camp-pops)",
    "Accept-Language": "en-US,en;q=0.9",
}

KNOWN_CAMPS = [
    "Ban Mai Nai Soi","Ban Mae Surin","Mae La Oon","Mae Ra Ma Luang",
    "Mae La","Umpiem Mai","Nupo","Ban Don Yang","Tham Hin"
]
ALIASES = {
    "ban mainai soi": "Ban Mai Nai Soi",
    "ban mai nai soi": "Ban Mai Nai Soi",
    "ban mae sur in": "Ban Mae Surin",
    "mae lao on": "Mae La Oon",
    "mae la oon": "Mae La Oon",
    "mae ra ma luang": "Mae Ra Ma Luang",
    "mae la": "Mae La",
    "umpiem mai": "Umpiem Mai",
    "um pie m": "Umpiem Mai",
    "nupo": "Nupo",
    "ban don yang": "Ban Don Yang",
    "tham hin": "Tham Hin",
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

def normalize(s):
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[^\S\r\n]+", " ", s)
    return s

def detect_category(text):
    if re.search(r"\bIDP\b|Internally Displaced", text, re.I):
        return "idp"
    return "refugee"

# --- Parsing ---
def parse_rows(text):
    rows = []
    clean = normalize(text)
    category = detect_category(clean)

    # known camps
    for camp in KNOWN_CAMPS:
        camp_pattern = re.sub(r" ", r"[ _-]?", re.escape(camp))
        pat = re.compile(rf"{camp_pattern}[^0-9]{{0,10}}([0-9][0-9,\.\s]{{2,}})", re.I)
        m = pat.search(clean)
        if m:
            num = re.sub(r"[^\d]", "", m.group(1))
            if num.isdigit():
                rows.append({
                    "camp_name": camp,
                    "population": int(num),
                    "category": category,
                    "parse_notes": "matched_known_camp",
                    "parse_confidence": 1.0
                })

    # generic fallback
    seen = {r["camp_name"] for r in rows}
    for m in re.finditer(r"([A-Z][A-Za-z /'\-]{2,40})\s+([0-9][0-9,\.\s]{2,})", clean):
        label = m.group(1).strip()
        digits = re.sub(r"[^\d]", "", m.group(2))
        if len(digits) < 3:
            continue
        norm = re.sub(r"[^a-z]", " ", label.lower()).strip()
        key = re.sub(r"\s+", " ", norm)
        if key in ALIASES:
            label = ALIASES[key]
        if label in seen:
            continue
        rows.append({
            "camp_name": label,
            "population": int(digits),
            "category": category,
            "parse_notes": "generic_label_number",
            "parse_confidence": 0.6 if label in ALIASES.values() else 0.5
        })
        seen.add(label)

    return rows

# --- Main ---
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
                # keep report_date sane for later sorting
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
            # 404 etc. — just skip
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

        rows = parse_rows(text) or [{
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

        # polite delay
        time.sleep(0.1)

    # Combine with existing & write
    df_new = pd.DataFrame.from_records(new_records)
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new

    if not combined.empty:
        with pd.option_context("mode.use_inf_as_na", True):
            combined["report_date"] = pd.to_datetime(combined["report_date"], errors="coerce")
        # drop exact duplicates
        combined = (combined
                    .drop_duplicates(subset=["report_date","camp_name","source_url","file_name","extract_method","parse_notes"],
                                     keep="last")
                    .sort_values(["report_date","camp_name"], na_position="last"))
        combined["report_date"] = combined["report_date"].dt.strftime("%Y-%m-01").f_]()_]()
