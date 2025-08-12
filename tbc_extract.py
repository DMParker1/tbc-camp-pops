#!/usr/bin/env python3
"""
Step 2: Download each source in data/derived/sources_index.csv and extract per-camp populations.
Outputs:
  - data/derived/tbc_camp_population_long.csv  (long/tidy: one row per camp per month)
  - data/derived/tbc_camp_population_wide.csv  (optional convenience pivot)
  - data/raw/<files>                            (cached artifacts)

Heuristics:
- Try PDF text (pdfplumber) first, then render to images (PyMuPDF) + OCR (Tesseract).
- For images (JPG/PNG), OCR directly.
- Match against the 9 Thai refugee camps; also capture other "Label 12345" pairs to catch IDP camps.
- Category detection is simple: if the page mentions "IDP" / "Internally Displaced", mark as "idp" else "refugee".

Env knobs (via workflow `env:`):
  - PYTHONUNBUFFERED: "1" to stream logs
"""

import os, io, re, sys, hashlib, time
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import requests

# ---- Config ----
RAW_DIR      = Path("data/raw")
DERIVED_DIR  = Path("data/derived")
INDEX_CSV    = DERIVED_DIR / "sources_index.csv"
OUT_LONG     = DERIVED_DIR / "tbc_camp_population_long.csv"
OUT_WIDE     = DERIVED_DIR / "tbc_camp_population_wide.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TBC-Extractor/1.0; +github.com/DMParker1/tbc-camp-pops)",
    "Accept-Language": "en-US,en;q=0.9",
}

KNOWN_CAMPS = [
    "Ban Mai Nai Soi","Ban Mae Surin","Mae La Oon","Mae Ra Ma Luang",
    "Mae La","Umpiem Mai","Nupo","Ban Don Yang","Tham Hin"
]
# Light alias map to help OCR quirks
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

# --- Utilities ---
def get(url, verify=True):
    """Requests GET with fallback to verify=False on SSL issues."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, verify=verify)
        r.raise_for_status()
        return r
    except requests.exceptions.SSLError:
        # retry with verify=False for this host only
        print(f"[warn] SSL verify failed for {url}; retrying without verification...", flush=True)
        r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        return r

def safe_filename(url, report_date):
    name = urlparse(url).path.split("/")[-1] or "file"
    # prefix date if present to reduce collisions
    if report_date:
        name = f"{report_date}_{name}"
    # sanitize minimally
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def download(url, report_date):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fname = safe_filename(url, report_date)
    out = RAW_DIR / fname
    if out.exists() and out.stat().st_size > 0:
        return out
    data = get(url).content
    out.write_bytes(data)
    return out

# --- Text extraction helpers ---
def extract_text_from_pdf(pdf_path):
    """Try pdfplumber text first; fallback to render+OCR."""
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
        # if the text is too sparse, fallback to OCR
        if len(combined.strip()) >= 200:
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
        pages = []
        for p in doc:
            pix = p.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
            img = ImageOps.autocontrast(img)
            img = img.filter(ImageFilter.SHARPEN)
            pages.append(pytesseract.image_to_string(img))
        return "\n".join(pages), method
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
    s = re.sub(r"[^\S\r\n]+", " ", s)  # collapse whitespace (except newlines)
    return s

def detect_category(text):
    if re.search(r"\bIDP\b|Internally Displaced", text, re.I):
        return "idp"
    return "refugee"

# --- Parsing ---
def parse_rows(text):
    """
    Return list of dicts with fields:
      camp_name, population, category, parse_notes, parse_confidence
    """
    rows = []
    clean = normalize(text)

    category = detect_category(clean)

    # 1) Try strict matches for known refugee camps
    for camp in KNOWN_CAMPS:
        # allow minor OCR noise around spaces/hyphens
        camp_pattern = re.sub(r" ", r"[ _-]?", re.escape(camp))
        # number can have commas/periods/spaces; don't capture trailing digits from page coords
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

    # 2) Generic "Label 12345" lines to catch IDP or unlabeled camps
    # Avoid duplicate of known camps captured above
    seen = {r["camp_name"] for r in rows}
    # tolerate labels with slashes/hyphens/spaces; keep it moderate to avoid legend text
    for m in re.finditer(r"([A-Z][A-Za-z /'\-]{2,40})\s+([0-9][0-9,\.\s]{2,})", clean):
        label = m.group(1).strip()
        digits = re.sub(r"[^\d]", "", m.group(2))
        if len(digits) < 3:
            continue
        # Map common OCR alias to known camp (if close)
        norm = re.sub(r"[^a-z]", " ", label.lower()).strip()
        label_key = re.sub(r"\s+", " ", norm)
        if label_key in ALIASES:
            label = ALIASES[label_key]
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

# --- Main pipeline ---
def main():
    if not INDEX_CSV.exists() or INDEX_CSV.stat().st_size == 0:
        print(f"[error] missing or empty {INDEX_CSV}", flush=True)
        sys.exit(0)  # graceful

    df_idx = pd.read_csv(INDEX_CSV, dtype=str).fillna("")
    if df_idx.empty:
        print("[warn] sources_index.csv has 0 rows", flush=True)
        # still write empty outputs with headers
        pd.DataFrame(columns=[
            "report_date","camp_name","population","category",
            "source_url","file_name","extract_method","parse_notes","parse_confidence"
        ]).to_csv(OUT_LONG, index=False)
        pd.DataFrame().to_csv(OUT_WIDE, index=False)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in df_idx.iterrows():
        report_date = (row.get("report_date") or "").strip()
        url         = (row.get("source_url")  or "").strip()
        fname       = (row.get("file_name")   or "").strip()

        if not url:
            continue

        try:
            local = download(url, report_date)
        except Exception as e:
            print(f"[warn] download failed: {url} -> {e}", flush=True)
            records.append({
                "report_date": report_date, "camp_name": None, "population": None, "category": None,
                "source_url": url, "file_name": fname, "extract_method": None,
                "parse_notes": f"download_error:{e}", "parse_confidence": 0.0
            })
            continue

        ext = local.suffix.lower()
        text, method = "", None
        try:
            if ext == ".pdf":
                text, method = extract_text_from_pdf(local)
            elif ext in (".jpg",".jpeg",".png"):
                text, method = extract_text_from_image(local)
            else:
                # try OCR anyway (some files lack extension)
                text, method = extract_text_from_image(local)
        except Exception as e:
            print(f"[warn] extraction failed: {local.name} -> {e}", flush=True)

        if not text.strip():
            records.append({
                "report_date": report_date, "camp_name": None, "population": None, "category": None,
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

        for r in rows:
            r.update({
                "report_date": report_date,
                "source_url": url,
                "file_name": local.name,
                "extract_method": method or "unknown",
            })
            records.append(r)

        # Be a good citizen to the host
        time.sleep(0.25)

    # Write outputs
    df_long = pd.DataFrame.from_records(records)
    # sort for readability
    if not df_long.empty:
        # Normalize date
        with pd.option_context("mode.use_inf_as_na", True):
            df_long["report_date"] = pd.to_datetime(df_long["report_date"], errors="coerce")
        df_long = df_long.sort_values(["report_date","camp_name"], na_position="last")
        df_long["report_date"] = df_long["report_date"].dt.strftime("%Y-%m-01").fillna("")

    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(OUT_LONG, index=False)

    # Make a simple wide pivot (latest value per month/camp if duplicates)
    try:
        if not df_long.empty:
            tmp = df_long.dropna(subset=["camp_name","population"]).copy()
            tmp["population"] = tmp["population"].astype(int)
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

    print(f"[done] wrote {OUT_LONG} ({len(df_long)} rows)", flush=True)

if __name__ == "__main__":
    main()
