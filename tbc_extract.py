#!/usr/bin/env python3
"""
tbc_extract.py — column-aware, line-locked extractor with series tagging.

Improvements:
- Same-line parsing: attach numbers only from the same OCR line as the camp label
  (or the immediate next line if the label wraps).
- Column selection: detect a header line and read numbers from the TBC/TBBC Assisted
  or MOI/UNHCR Verified 'Population' column. Preference for wide: tbc/tbbc > unhcr > unknown.
- Emits up to one row per (report_date, camp, series) per document.
- Keeps resume/caching, OCR, and env inputs.

Inputs via env (set in workflow):
  TBC_VERIFY_SSL ("true"/"false"), EXTRACT_SINCE (YYYY-MM-DD), EXTRACT_MAX_FILES,
  PROCESS_ORDER ("newest"|"oldest"), RESUME_MODE ("true"/"false"),
  OCR_DPI (int), OCR_PSM (int, default 6)
"""

import os, io, re, sys, time, logging
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# ---------- Paths ----------
RAW_DIR      = Path("data/raw")
DERIVED_DIR  = Path("data/derived")
INDEX_CSV    = DERIVED_DIR / "sources_index.csv"
OUT_LONG     = DERIVED_DIR / "tbc_camp_population_long.csv"
OUT_WIDE     = DERIVED_DIR / "tbc_camp_population_wide.csv"

# ---------- Env ----------
EXTRACT_SINCE     = os.getenv("EXTRACT_SINCE", "").strip()
EXTRACT_MAX_FILES = int(os.getenv("EXTRACT_MAX_FILES", "250"))
PROCESS_ORDER     = os.getenv("PROCESS_ORDER", "newest").lower()
RESUME_MODE       = os.getenv("RESUME_MODE", "true").lower() == "true"
OCR_DPI           = int(os.getenv("OCR_DPI", "200"))
OCR_PSM           = int(os.getenv("OCR_PSM", "6"))  # line-by-line

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TBC-Extractor/3.1; +github.com/DMParker1/tbc-camp-pops)",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------- Camps & Aliases ----------
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
    "mae ra ma luan g": "Mae Ra Ma Luang",
    "mae la": "Mae La",
    "umpiem mai": "Umpiem Mai",
    "um pie m": "Umpiem Mai",
    "nupo": "Nupo",
    "ban don yang": "Ban Don Yang",
    "tham hin": "Tham Hin",
    "thamhin": "Tham Hin",
}

GLOBAL_MIN = 50
GLOBAL_MAX = 300_000
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

SERIES_PREF = ["tbc", "tbbc", "unhcr", "unknown"]

# ---------- HTTP ----------
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

# ---------- OCR/Text ----------
def extract_text_from_pdf(pdf_path):
    txt_all = []
    method = "pdf-text"
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
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
            cfg = f"--psm {OCR_PSM} -l eng"
            parts.append(pytesseract.image_to_string(img, config=cfg))
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
        cfg = f"--psm {OCR_PSM} -l eng"
        txt = pytesseract.image_to_string(img, config=cfg)
        return txt, "img-ocr"
    except Exception as e:
        print(f"[warn] image OCR failed on {img_path.name}: {e}", flush=True)
        return "", "img-fail"

# ---------- Helpers ----------
def normalize(s):
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[^\S\r\n]+", " ", s)
    return s

def detect_category(text):
    if re.search(r"\bIDP\b|Internally Displaced", text, re.I):
        return "idp"
    return "refugee"

def build_camp_patterns():
    joiner = r"[ _\-]*"
    return {
        camp: re.compile(fr"\b{joiner.join(map(re.escape, camp.split()))}\b", re.I)
        for camp in KNOWN_CAMPS
    }

CAMP_PATS = build_camp_patterns()

def num_candidates(s):
    """
    Yield numeric candidates in string s as ints, filtering out plain 4-digit years
    (exactly 4 digits, no separators) and values out of global bounds.
    """
    for m in re.finditer(r"[0-9][0-9,\.\s]{1,12}", s):
        token = m.group(0).strip()
        digits = re.sub(r"[^\d]", "", token)
        if not digits:
            continue
        val = int(digits)
        if re.fullmatch(r"\d{4}", token) and 1900 <= val <= 2100:
            continue  # treat plain 4-digit tokens as years
        if not (GLOBAL_MIN <= val <= GLOBAL_MAX):
            continue
        yield val

def detect_series_by_header(header_line, col_start, col_end):
    seg = header_line[col_start:col_end].lower()
    if "tbbc" in seg:
        return "tbbc"
    if "tbc" in seg and "assisted" in seg:
        return "tbc"
    if "unhcr" in seg or "moi" in seg or "verified" in seg:
        return "unhcr"
    return "unknown"

def find_header_and_columns(lines):
    """
    Find a header line containing both Assisted and Verified population columns.
    Return (header_index, [(start,end,series), ...]) ordered left->right.
    """
    pat_assisted = re.compile(r"(TBC|TBBC).*?Assist", re.I)
    pat_verified = re.compile(r"(UNHCR|MOI).*?(Verify|Population)", re.I)

    for i, line in enumerate(lines[:60]):  # header near top
        if pat_assisted.search(line) and pat_verified.search(line):
            # rough spans: find header tokens, expand until next header token or end of line
            spans = []
            for m in re.finditer(r"(TBC.*?Assist.*?|TBBC.*?Assist.*?|UNHCR.*?Verify.*?|MOI.*?Verify.*?)", line, re.I):
                spans.append((m.start(), m.end()))
            if not spans:
                continue
            spans.sort()
            expanded = []
            for j, (s, e) in enumerate(spans):
                e2 = spans[j+1][0] if j+1 < len(spans) else len(line)
                expanded.append((s, e2))
            cols = []
            seen = set()
            for s, e in expanded:
                ser = detect_series_by_header(line, s, e)
                if ser in ("tbc","tbbc","unhcr") and ser not in seen:
                    cols.append((s, e, ser))
                    seen.add(ser)
            cols.sort(key=lambda x: x[0])
            return i, cols
    return None, []

def slice_by_cols(line, cols):
    return [(ser, line[max(0, s-2):min(len(line), e+2)]) for (s, e, ser) in cols]

def same_or_next_line(lines, i):
    line = lines[i]
    nxt = lines[i+1] if i+1 < len(lines) else ""
    return [line, nxt]

# ---------- Parsing ----------
def parse_rows(text):
    """
    Column-aware, line-locked parsing:
      - Detect header columns.
      - For each line containing a camp label, read numbers from each population column
        on the same line; if none, also check the next line (wrap).
      - Enforce camp bounds and build (camp, series) rows.
    """
    rows = []
    clean = normalize(text)
    category = detect_category(clean)
    lines = [ln.rstrip("\n") for ln in clean.splitlines() if ln.strip()]

    header_idx, cols = find_header_and_columns(lines)
    best = {}  # (camp, series) -> best value

    for i, line in enumerate(lines):
        for camp, cre in CAMP_PATS.items():
            if not cre.search(line):
                continue

            # try same line then next (wrap)
            found_any = False
            for candidate_line in same_or_next_line(lines, i):
                found_vals = []
                if cols:
                    for ser, seg in slice_by_cols(candidate_line, cols):
                        for val in num_candidates(seg):
                            found_vals.append((ser, val))
                else:
                    # no header: only numbers to the right of the camp label on this line
                    m = cre.search(candidate_line)
                    if m:
                        right = candidate_line[m.end():]
                        window = right[:80]
                        guess_ser = "unknown"
                        if re.search(r"\bTBC\b|\bTBBC\b", window, re.I):
                            guess_ser = "tbc"
                        if re.search(r"\bUNHCR\b|\bMOI\b", window, re.I):
                            guess_ser = "unhcr"
                        for val in num_candidates(window):
                            found_vals.append((guess_ser, val))

                if not found_vals:
                    continue

                lo, hi = CAMP_BOUNDS.get(camp, (GLOBAL_MIN, GLOBAL_MAX))
                for ser, val in found_vals:
                    if not (lo <= val <= hi):
                        continue
                    key = (camp, ser)
                    prev = best.get(key)
                    if (prev is None) or (val > prev):
                        best[key] = val
                found_any = True
                if found_any:
                    break  # move to next camp

    for (camp, ser), val in best.items():
        rows.append({
            "camp_name": camp,
            "population": int(val),
            "category": category,
            "series": ser,
            "parse_notes": "line_locked_column",
            "parse_confidence": 0.97 if ser in ("tbc","tbbc","unhcr") else 0.9,
        })

    return rows

# ---------- Main ----------
def main():
    if not INDEX_CSV.exists() or INDEX_CSV.stat().st_size == 0:
        print(f"[error] missing or empty {INDEX_CSV}", flush=True)
        sys.exit(0)

    df_idx = pd.read_csv(INDEX_CSV, dtype=str).fillna("")
    if df_idx.empty:
        print("[warn] sources_index.csv has 0 rows", flush=True)
        pd.DataFrame(columns=[
            "report_date","camp_name","population","category","series",
            "source_url","file_name","extract_method","parse_notes","parse_confidence"
        ]).to_csv(OUT_LONG, index=False)
        pd.DataFrame().to_csv(OUT_WIDE, index=False)
        return

    with pd.option_context("mode.use_inf_as_na", True):
        df_idx["report_date"] = pd.to_datetime(df_idx["report_date"], errors="coerce")

    if EXTRACT_SINCE:
        try:
            since = pd.to_datetime(EXTRACT_SINCE)
            df_idx = df_idx[df_idx["report_date"] >= since]
        except Exception as e:
            print(f"[warn] bad EXTRACT_SINCE={EXTRACT_SINCE}: {e}", flush=True)

    # Resume: skip files already in LONG
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

    # Order & limit
    df_idx = df_idx.dropna(subset=["source_url"])
    if PROCESS_ORDER == "oldest":
        df_idx = df_idx.sort_values(["report_date", "file_name"], na_position="last")
    else:
        df_idx = df_idx.sort_values(["report_date", "file_name"], na_position="last", ascending=[False, True])

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
                "series": "unknown",
                "source_url": url, "file_name": local.name, "extract_method": method or "none",
                "parse_notes": "no_text_extracted", "parse_confidence": 0.0
            })
            continue

        rows = parse_rows(text)
        if not rows:
            rows = [{
                "camp_name": None, "population": None, "category": None,
                "series": "unknown",
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

        time.sleep(0.05)  # polite

    # ---------- Write LONG ----------
    df_new = pd.DataFrame.from_records(new_records)
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new

    if not combined.empty:
        with pd.option_context("mode.use_inf_as_na", True):
            combined["report_date"] = pd.to_datetime(combined["report_date"], errors="coerce")
        combined = (combined
                    .drop_duplicates(
                        subset=["report_date","camp_name","series","source_url","file_name","extract_method","parse_notes"],
                        keep="last")
                    .sort_values(["report_date","camp_name","series"], na_position="last"))
        combined["report_date"] = combined["report_date"].dt.strftime("%Y-%m-01").fillna("")

    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_LONG, index=False)

    # ---------- Build preferred-series WIDE ----------
    try:
        if not combined.empty:
            tmp = combined.dropna(subset=["camp_name","population"]).copy()
            tmp["population"] = pd.to_numeric(tmp["population"], errors="coerce")
            tmp = tmp[tmp["population"].between(GLOBAL_MIN, GLOBAL_MAX)]

            def within_bounds(row):
                camp = str(row["camp_name"])
                val  = row["population"]
                if camp in CAMP_BOUNDS and pd.notna(val):
                    lo, hi = CAMP_BOUNDS[camp]
                    return lo <= val <= hi
                return True
            tmp = tmp[tmp.apply(within_bounds, axis=1)]

            tmp["series"] = tmp["series"].fillna("unknown").str.lower()
            tmp["series_rank"] = tmp["series"].map({s:i for i,s in enumerate(SERIES_PREF)}).fillna(len(SERIES_PREF)).astype(int)

            with pd.option_context("mode.use_inf_as_na", True):
                tmp["report_date"] = pd.to_datetime(tmp["report_date"], errors="coerce")

            tmp = (tmp
                   .sort_values(["report_date","camp_name","series_rank","population"], ascending=[True, True, True, False])
                   .drop_duplicates(["report_date","camp_name"], keep="first"))

            wide = (tmp
                    .pivot(index="report_date", columns="camp_name", values="population")
                    .sort_index())
            for c in wide.columns:
                wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("Int64")
            wide.to_csv(OUT_WIDE)
        else:
            pd.DataFrame().to_csv(OUT_WIDE, index=False)
    except Exception as e:
        print(f"[warn] could not generate wide CSV: {e}", flush=True)
        pd.DataFrame().to_csv(OUT_WIDE, index=False)

    print(f"[done] wrote {OUT_LONG} (+{len(df_new)} new rows, total {len(combined)})", flush=True)

if __name__ == "__main__":
    main()
