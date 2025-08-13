#!/usr/bin/env python3
"""
TBC camp-populations extractor — geometry-locked, header-anchored parser

What this version adds/fixes
- **Series split is reliable**: columns are anchored to detected headers (TBC/TBBC Verified/Feeding/Assisted vs MOI/UNHCR Verified/Population) using x‑spans.
- **Mae La vs Mae La Oon**: camp rows are segmented from camp-label bounding boxes and numbers are accepted only if their center‑y lies inside that row's y‑band (prevents row bleed and name collisions).
- **Subseries preserved**: writes `subseries` (verified|feeding|assisted|—) in the long file; preferred wide still follows TBC/TBBC > UNHCR.
- **Date bounds**: honors `EXTRACT_SINCE` and `EXTRACT_UNTIL` (env vars) before work selection.
- **Dual audit pivot**: also emits `tbc_camp_population_wide_dual.csv` with both series side‑by‑side for QA.

Dependencies (ensure in requirements.txt):
  pdfplumber, unidecode, pandas, requests, Pillow, pytesseract, PyMuPDF (optional OCR path), tabulate>=0.9
"""

import os, io, re, sys, time, logging
from pathlib import Path
from urllib.parse import urlparse
import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import urllib3
from unidecode import unidecode

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# ---------- Paths ----------
RAW_DIR      = Path("data/raw")
DERIVED_DIR  = Path("data/derived")
INDEX_CSV    = DERIVED_DIR / "sources_index.csv"
OUT_LONG     = DERIVED_DIR / "tbc_camp_population_long.csv"
OUT_WIDE     = DERIVED_DIR / "tbc_camp_population_wide.csv"
OUT_WIDE_DUAL= DERIVED_DIR / "tbc_camp_population_wide_dual.csv"

# ---------- Env ----------
EXTRACT_SINCE     = os.getenv("EXTRACT_SINCE", "").strip()
EXTRACT_UNTIL     = os.getenv("EXTRACT_UNTIL", "").strip()
EXTRACT_MAX_FILES = int(os.getenv("EXTRACT_MAX_FILES", "250"))
PROCESS_ORDER     = os.getenv("PROCESS_ORDER", "newest").lower()
RESUME_MODE       = os.getenv("RESUME_MODE", "true").lower() == "true"
OCR_DPI           = int(os.getenv("OCR_DPI", "200"))
OCR_PSM           = int(os.getenv("OCR_PSM", "6"))  # line-by-line

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TBC-Extractor/4.0; +github.com/DMParker1/tbc-camp-pops)",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------- Camps & bounds ----------
DEFAULT_CANON = [
    "Mae Ra Ma Luang", "Mae La Oon", "Umpiem Mai",
    "Ban Don Yang", "Tham Hin", "Nupo", "Nu Po", "Mae La",
    "Ban Mai Nai Soi", "Ban Mae Surin", "Shoklo",
    "Site 1", "Site 2", "Site 3"
]
DEFAULT_SYNS = {
    "Mae La Oon": ["Mae La Oon", "Mae La-Oon", "Mae LaOon"],
    "Mae Ra Ma Luang": ["Mae Ra Ma Luang", "Mae Rama Luang", "Mae Ra Mat Luang"],
    "Umpiem Mai": ["Umpiem", "Umphiem", "Umpiem Mai"],
    "Nu Po": ["Nu Po", "Nupo"],
    "Nupo": ["Nu Po", "Nupo"],
    "Ban Don Yang": ["Ban Don Yang", "Baan Don Yang", "Ban Donyang"],
    "Tham Hin": ["Tham Hin", "Tam Hin"],
    "Mae La": ["Mae La", "Maela"],
    "Ban Mai Nai Soi": ["Ban Mai Nai Soi"],
    "Ban Mae Surin": ["Ban Mae Surin"],
    "Shoklo": ["Shoklo", "Sho Klo"],
}

GLOBAL_MIN = 50
GLOBAL_MAX = 300_000
CAMP_BOUNDS = {
    "Mae La":          (10_000,  70_000),
    "Umpiem Mai":      (3_000,   45_000),
    "Nu Po":           (2_000,   40_000),
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

# ---------- Text & OCR ----------
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
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", unidecode((s or "").lower())).strip()

def normalize_whitespace(s):
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[^\S\r\n]+", " ", s)
    return s

def detect_category(text):
    if re.search(r"\bIDP\b|Internally Displaced", text, re.I):
        return "idp"
    return "refugee"

# Optional, date-aware lineage file
LINEAGE_CSV = Path("data/reference/camp_lineage.csv")

def load_lineage() -> Tuple[List[str], Dict[str, List[str]], Dict[str, Tuple[Optional[dt.date], Optional[dt.date]]]]:
    if not LINEAGE_CSV.exists():
        canon = sorted(DEFAULT_CANON, key=len, reverse=True)
        syns = DEFAULT_SYNS
        active = {}
        return canon, syns, active
    df = pd.read_csv(LINEAGE_CSV)
    canon, syns, active = [], {}, {}
    for _, r in df.iterrows():
        c = str(r.get("canonical", "")).strip()
        if not c:
            continue
        canon.append(c)
        al = [x.strip() for x in str(r.get("aliases", "")).split(";") if x.strip()]
        syns[c] = [c] + al
        st = str(r.get("start", "")).strip() or None
        en = str(r.get("end", "")).strip() or None
        st_d = dt.datetime.strptime(st, "%Y-%m-%d").date() if st else None
        en_d = dt.datetime.strptime(en, "%Y-%m-%d").date() if en else None
        active[c] = (st_d, en_d)
    canon = sorted(canon, key=len, reverse=True)
    return canon, syns, active

def is_active(camp: str, report_date: dt.date, active_map: Dict[str, Tuple[Optional[dt.date], Optional[dt.date]]]):
    if camp not in active_map:
        return True
    st, en = active_map[camp]
    if st and report_date < st:
        return False
    if en and report_date > en:
        return False
    return True

# ---------- Geometry parsing (pdfplumber) ----------
NUM_RE = re.compile(r"^[\s]*[0-9][0-9,]*[\s]*$")

HEADER_PATTERNS = {
    "tbbc_verified": re.compile(r"(?:\bTBC\b|\bTBBC\b|\bBBC\b).*verified", re.I),
    "tbbc_feeding":  re.compile(r"(?:\bTBC\b|\bTBBC\b|\bBBC\b).*(feeding|assisted)", re.I),
    "unhcr":         re.compile(r"(?:\bMOI/)?\bUNHCR\b.*(population|verified)?", re.I),
}

def _group_lines(words, y_tol=2):
    lines = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        placed = False
        for L in lines:
            if abs(L["top"] - w["top"]) <= y_tol:
                L["words"].append(w)
                L["top"] = min(L["top"], w["top"])  # tighten
                L["bottom"] = max(L["bottom"], w["bottom"])  # expand
                placed = True
                break
        if not placed:
            lines.append({"top": w["top"], "bottom": w["bottom"], "words": [w]})
    for L in lines:
        L["x0"] = min(w["x0"] for w in L["words"])
        L["x1"] = max(w["x1"] for w in L["words"])
        L["text"] = " ".join(w["text"] for w in sorted(L["words"], key=lambda w: w["x0"]))
    return lines

def detect_columns(page) -> Dict[str, Tuple[float, float]]:
    words = page.extract_words(use_text_flow=True, y_tolerance=2, x_tolerance=1, keep_blank_chars=False)
    lines = _group_lines(words)
    spans = {}
    for L in lines[:80]:  # headers live near the top
        t = L["text"]
        for key, pat in HEADER_PATTERNS.items():
            if pat.search(t):
                if key not in spans:
                    spans[key] = [L["x0"], L["x1"]]
                else:
                    spans[key][0] = min(spans[key][0], L["x0"])  # widen
                    spans[key][1] = max(spans[key][1], L["x1"])
    # flatten -> (x0,x1)
    spans = {k: (v[0], v[1]) for k, v in spans.items()}
    return spans

def center_x(w):
    return (w["x0"] + w["x1"]) / 2.0

def center_y(w):
    return (w["top"] + w["bottom"]) / 2.0

def match_camp_from_text(text: str, canon_ordered: List[str], syns: Dict[str, List[str]]) -> Optional[str]:
    lab = _norm(text)
    for canon in canon_ordered:
        for v in syns.get(canon, [canon]):
            if re.search(rf"\b{re.escape(_norm(v))}\b", lab):
                return canon
    return None

def segment_rows(page, canon_ordered, syns, report_date: dt.date, active_map, y_pad=2):
    words = page.extract_words(use_text_flow=True, y_tolerance=2, x_tolerance=1, keep_blank_chars=False)
    lines = _group_lines(words)
    rows = []
    claimed = []  # y-span list to avoid duplicates
    for L in lines:
        camp = match_camp_from_text(L["text"], canon_ordered, syns)
        if not camp:
            continue
        if not is_active(camp, report_date, active_map):
            continue
        y0, y1 = L["top"] - y_pad, L["bottom"] + y_pad
        # skip if overlapping a previously claimed span (longest-name-first ordering already helps)
        overlap = any(not (y1 < a or y0 > b) for (a, b) in claimed)
        if overlap:
            continue
        claimed.append((y0, y1))
        rows.append({"camp": camp, "y0": y0, "y1": y1, "x0": L["x0"], "x1": L["x1"]})
    rows.sort(key=lambda r: (r["y0"], r["camp"]))
    return rows

def extract_values_from_page(page, report_date: dt.date, source_url: str, cols: Dict[str, Tuple[float, float]], rows):
    words = page.extract_words(use_text_flow=True, y_tolerance=2, x_tolerance=1, keep_blank_chars=False)
    numerics = [w for w in words if NUM_RE.match(w["text"])]
    out = []
    for row in rows:
        y0, y1 = row["y0"], row["y1"]
        in_row = [w for w in numerics if y0 <= center_y(w) <= y1]
        for w in in_row:
            cx = center_x(w)
            series, subseries = None, ""
            if "tbbc_verified" in cols and cols["tbbc_verified"][0] <= cx <= cols["tbbc_verified"][1]:
                series, subseries = "tbbc", "verified"
            elif "tbbc_feeding" in cols and cols["tbbc_feeding"][0] <= cx <= cols["tbbc_feeding"][1]:
                series, subseries = "tbbc", "feeding"
            elif "unhcr" in cols and cols["unhcr"][0] <= cx <= cols["unhcr"][1]:
                series, subseries = "unhcr", "verified"
            else:
                continue
            val = int(re.sub(r"[^\d]", "", w["text"]))
            # sanity/bounds
            lo, hi = CAMP_BOUNDS.get(row["camp"], (GLOBAL_MIN, GLOBAL_MAX))
            if not (lo <= val <= hi):
                continue
            out.append({
                "date": report_date.strftime("%Y-%m-%d"),
                "camp": row["camp"],
                "series": series,
                "subseries": subseries,
                "value": val,
                "source_url": source_url,
                "parse_notes": "geom_header_locked",
                "parse_confidence": 0.98,
            })
    return out

def parse_pdf_geometry(pdf_path: Path, report_date: dt.date, source_url: str) -> List[Dict]:
    import pdfplumber
    canon, syns, active_map = load_lineage()
    all_rows: List[Dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            cols = detect_columns(page)
            if not cols:
                continue
            rows = segment_rows(page, canon, syns, report_date, active_map)
            if not rows:
                continue
            rows_out = extract_values_from_page(page, report_date, source_url, cols, rows)
            all_rows.extend(rows_out)
    return all_rows

# ---------- Fallback (text-based) ----------
# Conservative fallback to keep older behavior when geometry isn't available
JOINER = r"[ _\-]*"
KNOWN_CAMPS_FALLBACK = [
    "Ban Mai Nai Soi","Ban Mae Surin","Mae La Oon","Mae Ra Ma Luang",
    "Mae La","Umpiem Mai","Nupo","Nu Po","Ban Don Yang","Tham Hin"
]
CAMP_PATS_FALLBACK = {
    camp: re.compile(fr"\b{JOINER.join(map(re.escape, camp.split()))}\b", re.I)
    for camp in KNOWN_CAMPS_FALLBACK
}

def num_candidates(s):
    for m in re.finditer(r"[0-9][0-9,\.\s]{1,12}", s):
        token = m.group(0).strip()
        digits = re.sub(r"[^\d]", "", token)
        if not digits:
            continue
        val = int(digits)
        if re.fullmatch(r"\d{4}", token) and 1900 <= val <= 2100:
            continue
        if not (GLOBAL_MIN <= val <= GLOBAL_MAX):
            continue
        yield val

def detect_series_by_header(header_line, s, e):
    seg = header_line[s:e].lower()
    if "tbbc" in seg:
        return "tbbc"
    if "tbc" in seg and "assist" in seg:
        return "tbc"
    if "unhcr" in seg or "moi" in seg or "verif" in seg:
        return "unhcr"
    return "unknown"

def find_header_and_columns(lines):
    """
    Locate header(s) that declare TBBC/TBC Assisted/Feeding and UNHCR/MOI Verified/Population.
    Now looks across a 3-line window to handle layouts where:
        TBBC
        Verified Caseload   Feeding Figure   MOI/ UNHCR Population
    Returns: (header_idx, [(start,end,series), ...]) ordered left->right
    """
    pat_any_tbbc   = re.compile(r"(?:\bTBC\b|\bTBBC\b|\bBBC\b)", re.I)
    pat_verified   = re.compile(r"verified(\s+caseload)?", re.I)
    pat_feeding    = re.compile(r"(feeding\s+figure|feeding|assisted)", re.I)
    pat_unhcr      = re.compile(r"(?:\bMOI\b[^A-Za-z0-9/]*\/\s*)?\bUNHCR\b", re.I)
    pat_unhcr_aux  = re.compile(r"(population|verified)", re.I)

    # Examine the first ~80 lines (typical header area), but use a 3-line rolling window
    N = min(80, len(lines))
    for i in range(N):
        win = lines[i:i+3]
        if not win:
            continue
        window_text = " ".join(win)
        # If the window does not look like a header, keep scanning
        if not ((pat_any_tbbc.search(window_text) and (pat_verified.search(window_text) or pat_feeding.search(window_text)))
                or (pat_unhcr.search(window_text) and pat_unhcr_aux.search(window_text))):
            continue

        # Find left-to-right spans within the window_text
        spans = []
        for m in re.finditer(r"(?:TBC|TBBC|BBC).*?(?:Assist|Feeding)", window_text, re.I):
            spans.append((m.start(), m.end(), "tbc_tbbc_assist"))
        for m in re.finditer(r"(?:UNHCR|MOI).*?(?:Verify|Population)", window_text, re.I):
            spans.append((m.start(), m.end(), "unhcr"))

        if not spans:
            # Fall back to separate cues for TBBC verified/feeding if the above didn’t match
            if pat_verified.search(window_text):
                spans.append((0, len(window_text), "tbc_tbbc_assist"))
            if pat_unhcr.search(window_text) and pat_unhcr_aux.search(window_text):
                spans.append((0, len(window_text), "unhcr"))

        if not spans:
            continue

        spans.sort(key=lambda x: x[0])

        # Expand each span to the next span’s start (or end of line) so we get column bands
        expanded = []
        for j, (s, e, tag) in enumerate(spans):
            e2 = spans[j+1][0] if j+1 < len(spans) else len(window_text)
            expanded.append((s, e2, tag))

        # Map tags to the series keys we use downstream, de-duplicated and ordered left->right
        cols, seen = [], set()
        for s, e, tag in expanded:
            if tag == "unhcr":
                ser = "unhcr"
            else:
                # This column holds TBBC/TBC numbers (verified/feeding will be separated later)
                ser = "tbc"  # upstream code already handles 'tbbc' synonymically
            if ser not in seen:
                cols.append((s, e, ser))
                seen.add(ser)

        if cols:
            return i, cols

    return None, []

def slice_by_cols(line, cols):
    return [(ser, line[max(0, s-3):min(len(line), e+3)]) for (s, e, ser) in cols]

def same_or_next_line(lines, i):
    line = lines[i]
    nxt = lines[i+1] if i+1 < len(lines) else ""
    return [line, nxt]

def parse_rows_text(text):
    rows = []
    clean = normalize_whitespace(text)
    category = detect_category(clean)
    lines = [ln.rstrip("\n") for ln in clean.splitlines() if ln.strip()]
    header_idx, cols = find_header_and_columns(lines)
    best = {}
    for i, line in enumerate(lines):
        for camp, cre in CAMP_PATS_FALLBACK.items():
            if not cre.search(line):
                continue
            for candidate_line in same_or_next_line(lines, i):
                found_vals = []
                if cols:
                    for ser, seg in slice_by_cols(candidate_line, cols):
                        for val in num_candidates(seg):
                            found_vals.append((ser, val))
                else:
                    m = cre.search(candidate_line)
                    if m:
                        right = candidate_line[m.end():][:80]
                        ser = "unknown"
                        if re.search(r"\bTBC\b|\bTBBC\b", right, re.I): ser = "tbc"
                        if re.search(r"\bUNHCR\b|\bMOI\b", right, re.I): ser = "unhcr"
                        for val in num_candidates(right):
                            found_vals.append((ser, val))
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
                if any(k[0] == camp for k in best):
                    break
    for (camp, ser), val in best.items():
        sub = "verified" if ser in ("tbc","tbbc") else ("verified" if ser=="unhcr" else "")
        rows.append({
            "camp": camp,
            "series": ser,
            "subseries": sub,
            "value": int(val),
            "parse_notes": "text_line_locked",
            "parse_confidence": 0.90 if ser in ("tbc","tbbc","unhcr") else 0.85,
        })
    return rows

# ---------- Main pipeline ----------

def main():
    if not INDEX_CSV.exists() or INDEX_CSV.stat().st_size == 0:
        print(f"[error] missing or empty {INDEX_CSV}", flush=True)
        sys.exit(0)

    df_idx = pd.read_csv(INDEX_CSV, dtype=str).fillna("")
    if df_idx.empty:
        print("[warn] sources_index.csv has 0 rows", flush=True)
        pd.DataFrame(columns=[
            "report_date","camp_name","population","category","series","subseries",
            "source_url","file_name","extract_method","parse_notes","parse_confidence"
        ]).to_csv(OUT_LONG, index=False)
        pd.DataFrame().to_csv(OUT_WIDE, index=False)
        pd.DataFrame().to_csv(OUT_WIDE_DUAL, index=False)
        return

    with pd.option_context("mode.use_inf_as_na", True):
        df_idx["report_date"] = pd.to_datetime(df_idx["report_date"], errors="coerce")

    if EXTRACT_SINCE:
        try:
            since = pd.to_datetime(EXTRACT_SINCE)
            df_idx = df_idx[df_idx["report_date"] >= since]
        except Exception as e:
            print(f"[warn] bad EXTRACT_SINCE={EXTRACT_SINCE}: {e}", flush=True)

    if EXTRACT_UNTIL:
        try:
            until = pd.to_datetime(EXTRACT_UNTIL)
            df_idx = df_idx[df_idx["report_date"] <= until]
        except Exception as e:
            print(f"[warn] bad EXTRACT_UNTIL={EXTRACT_UNTIL}: {e}", flush=True)

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

    df_idx = df_idx.dropna(subset=["source_url"])
    if PROCESS_ORDER == "oldest":
        df_idx = df_idx.sort_values(["report_date", "file_name"], na_position="last")
    else:
        df_idx = df_idx.sort_values(["report_date", "file_name"], na_position="last", ascending=[False, True])

    candidates = []
    for _, r in df_idx.iterrows():
        key = (str(r.get("source_url", "")), str(r.get("file_name", "")))
        if RESUME_MODE and key in processed_keys:
            continue
        candidates.append(r)
        if len(candidates) >= EXTRACT_MAX_FILES:
            break

    print(f"[info] extracting from {len(candidates)} new files "
          f"(resume={RESUME_MODE}, since='{EXTRACT_SINCE or 'ALL'}', until='{EXTRACT_UNTIL or 'LATEST'}', order={PROCESS_ORDER})",
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
            elif ext in (".jpg", ".jpeg", ".png"):
                text, method = extract_text_from_image(local)
            else:
                text, method = extract_text_from_image(local)
        except Exception as e:
            print(f"[warn] extraction failed: {local.name} -> {e}", flush=True)

        # First try GEOMETRY on the original PDF bytes (only possible for PDFs)
        geom_rows: List[Dict] = []
        if local.suffix.lower() == ".pdf":
            try:
                rd_date = dt.datetime.strptime(rd_str, "%Y-%m-%d").date() if rd_str else None
                if rd_date is not None:
                    geom_rows = parse_pdf_geometry(local, rd_date, url)
            except Exception as e:
                print(f"[warn] geometry parse failed on {local.name}: {e}", flush=True)

        if geom_rows:
            # we have series+subseries with geometry lock
            category = detect_category(text or "")
            for row in geom_rows:
                new_records.append({
                    "report_date": rd_str,
                    "camp_name": row["camp"],
                    "population": row["value"],
                    "category": category,
                    "series": row["series"],
                    "subseries": row.get("subseries", ""),
                    "source_url": url,
                    "file_name": local.name,
                    "extract_method": method or "unknown",
                    "parse_notes": row.get("parse_notes", "geom_header_locked"),
                    "parse_confidence": row.get("parse_confidence", 0.98),
                })
            time.sleep(0.05)
            continue  # next file

        # Fallback: text-line parsing
        if not (text or "").strip():
            new_records.append({
                "report_date": rd_str, "camp_name": None, "population": None, "category": None,
                "series": "unknown", "subseries": "",
                "source_url": url, "file_name": local.name, "extract_method": method or "none",
                "parse_notes": "no_text_extracted", "parse_confidence": 0.0
            })
            continue

        rows_text = parse_rows_text(text)
        if not rows_text:
            rows_text = [{
                "camp": None, "series": "unknown", "subseries": "", "value": None,
                "parse_notes": "no_rows_parsed", "parse_confidence": 0.0
            }]
        for row in rows_text:
            new_records.append({
                "report_date": rd_str,
                "camp_name": row.get("camp"),
                "population": row.get("value"),
                "category": detect_category(text or ""),
                "series": row.get("series", "unknown"),
                "subseries": row.get("subseries", ""),
                "source_url": url,
                "file_name": local.name,
                "extract_method": method or "unknown",
                "parse_notes": row.get("parse_notes", "text_line_locked"),
                "parse_confidence": row.get("parse_confidence", 0.9),
            })
        time.sleep(0.05)

    # ---------- Write LONG ----------
    df_new = pd.DataFrame.from_records(new_records)
    if RESUME_MODE and OUT_LONG.exists() and OUT_LONG.stat().st_size:
        try:
            existing = pd.read_csv(OUT_LONG, dtype=str).fillna("")
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, df_new], ignore_index=True) if not df_new.empty else existing

    if not combined.empty:
        with pd.option_context("mode.use_inf_as_na", True):
            combined["report_date"] = pd.to_datetime(combined["report_date"], errors="coerce")
        # IMPORTANT: include subseries in dedupe so verified vs feeding both survive
        combined = (combined
                    .drop_duplicates(
                        subset=["report_date","camp_name","series","subseries","source_url","file_name","extract_method","parse_notes"],
                        keep="last")
                    .sort_values(["report_date","camp_name","series","subseries"], na_position="last"))
        combined["report_date"] = combined["report_date"].dt.strftime("%Y-%m-01").fillna("")

    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_LONG, index=False)

    # ---------- Preferred-series WIDE ----------
    try:
        if not combined.empty:
            tmp = combined.dropna(subset=["camp_name","population"]).copy()
            tmp["population"] = pd.to_numeric(tmp["population"], errors="coerce")
            tmp = tmp[tmp["population"].between(GLOBAL_MIN, GLOBAL_MAX)]

            def within_bounds(row):
                camp = str(row["camp_name"]) if pd.notna(row["camp_name"]) else ""
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

    # ---------- Dual-series WIDE (audit) ----------
    try:
        if not combined.empty:
            dual = combined.dropna(subset=["camp_name","population","series"]).copy()
            dual["population"] = pd.to_numeric(dual["population"], errors="coerce")
            dual = dual[dual["population"].between(GLOBAL_MIN, GLOBAL_MAX)]
            dual["col"] = dual["report_date"] + "|" + dual["series"].str.lower()
            dual_w = dual.pivot_table(index="camp_name", columns="col", values="population", aggfunc="first")
            dual_w.to_csv(OUT_WIDE_DUAL)
        else:
            pd.DataFrame().to_csv(OUT_WIDE_DUAL, index=False)
    except Exception as e:
        print(f"[warn] could not generate dual wide CSV: {e}", flush=True)
        pd.DataFrame().to_csv(OUT_WIDE_DUAL, index=False)

    print(f"[done] wrote {OUT_LONG} (+{len(df_new)} new rows, total {len(combined)})", flush=True)

if __name__ == "__main__":
    main()
