#!/usr/bin/env python3
"""
TBC camp-populations extractor — fast geometry-first drop-in

Purpose
- Keep runs fast by trying pdf geometry (header-anchored) first.
- Skip OCR entirely unless explicitly enabled via env.
- Preserve your long + wide outputs and the preferred-series logic.

Env knobs (defaults shown)
  EXTRACT_SINCE       = ""            # YYYY-MM-DD lower bound (inclusive)
  EXTRACT_UNTIL       = ""            # YYYY-MM-DD upper bound (inclusive)
  EXTRACT_MAX_FILES   = "250"         # max files processed this run
  PROCESS_ORDER       = "newest"      # or "oldest"
  RESUME_MODE         = "true"        # skip already-seen files/rows
  OCR_DPI             = "200"         # only used when OCR enabled
  DISABLE_OCR         = "true"        # fast path; set false only when needed
  OCR_PSM             = "6"           # tesseract page segmentation mode when OCR enabled
  SKIP_CRAWL          = "true"        # reuse sources_index.csv if available for fast iteration

Behavior
- Geometry-first (header-anchored). If geometry finds no UNHCR, fill via embedded text (no OCR).
- TBBC spans clamped to the left of UNHCR.
- NEW: if geometry finds no TBBC, also supplement via embedded text (no OCR).
- NEW: more permissive numeric capture (handles *, †, () around numbers).

Outputs
- data/derived/tbc_camp_population_long.csv
- data/derived/tbc_camp_population_wide.csv
- data/derived/tbc_camp_population_wide_dual.csv
"""

import os, sys, re, io, time, math, json, shutil, hashlib, tempfile, logging, zipfile, random
import datetime as dt
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from urllib.parse import urlparse
import requests
import pandas as pd
import numpy as np

# quiet noisy libs
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
OUT_WIDE_DUAL= DERIVED_DIR / "tbc_camp_population_wide_dual.csv"

# ---------- Env ----------
EXTRACT_SINCE        = os.environ.get("EXTRACT_SINCE", "").strip()
EXTRACT_UNTIL        = os.environ.get("EXTRACT_UNTIL", "").strip()
EXTRACT_MAX_FILES    = int(os.environ.get("EXTRACT_MAX_FILES", "250"))
PROCESS_ORDER        = os.environ.get("PROCESS_ORDER", "newest").strip().lower()
RESUME_MODE          = os.environ.get("RESUME_MODE", "true").strip().lower() == "true"
DISABLE_OCR          = os.environ.get("DISABLE_OCR", "true").strip().lower() == "true"
OCR_DPI              = int(os.environ.get("OCR_DPI", "200"))
OCR_PSM              = int(os.environ.get("OCR_PSM", "6"))
SKIP_CRAWL           = os.environ.get("SKIP_CRAWL", "true").strip().lower() == "true"

# ---------- Globals / reference ----------
GLOBAL_MIN = 100
GLOBAL_MAX = 200_000

CANONICAL_ORDER = [
    "Ban Mai Nai Soi","Ban Mae Surin","Mae La Oon","Mae Ra Ma Luang",
    "Mae La","Umpiem Mai", "Nu Po","Ban Don Yang","Tham Hin"
]
CAMP_ALIASES = {
    "Nu Po": ["Nupo","Nu  Po","Nu  Po"],
    "Nupo": ["Nu Po","Nu  Po","Nupo"],
    "Mae La Oon": ["Mae La Oon","Mae La Oon (MLA)","Mae  La  Oon"],
    "Mae Ra Ma Luang": ["Mae Ra Ma Luang","Mae  Ra  Ma  Luang","Mae Ra Ma Laung"],
    "Umpiem Mai": ["Umphiem","Umpiem","Umpiem  Mai","Umphiem Mai"],
    "Ban Mae Surin": ["BMS","Ban Mae Surin  (BMS)","Mae Surin"],
    "Ban Mai Nai Soi": ["BMN","Ban Mai Nai Soi  (BMN)","Mai Nai Soi"],
}
ACTIVE_MAP = {
    # if needed, (start,end) bounds; left as open for now; can be filled via lineage file
}
# --- Camp name normalizer (aliases -> canonical) ---
def _build_canon_map():
    m = {}
    # map canonicals to themselves
    for canon in CANONICAL_ORDER:
        m[canon.lower()] = canon
    # map each alias to its canonical
    for canon, aliases in CAMP_ALIASES.items():
        m[canon.lower()] = canon
        for a in aliases:
            m[a.lower()] = canon
    return m

_CANON_MAP = _build_canon_map()

def norm_camp(name: str) -> str:
    if not name:
        return name
    return _CANON_MAP.get(str(name).strip().lower(), name)
CANON_BOUNDS = {
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

# ---------- Utilities ----------
def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).replace("\u00a0", " ")

def detect_category(text: str) -> str:
    # keep as simple tag for now
    return "refugee"

def download(url: str, report_date_str: str) -> Optional[Path]:
    """Download to RAW_DIR organized by year-month; returns local file path."""
    try:
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix or ".pdf"
        folder = RAW_DIR / report_date_str[:7]
        folder.mkdir(parents=True, exist_ok=True)
        fn = f"{sha1(url)}{ext}"
        dest = folder / fn
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        r = requests.get(url, timeout=30, verify=False)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        return dest
    except Exception as e:
        print(f"[warn] download failed for {url}: {e}", flush=True)
        return None

def load_lineage():
    canon = set(CANONICAL_ORDER)
    syns  = dict(CAMP_ALIASES)
    active = ACTIVE_MAP

    ref = Path("data/reference/camp_lineage.csv")
    rules = []   # list of dicts per row
    merges = {}  # canonical -> (merge_into, merge_start_date)

    def _d(s):
        s = (s or "").strip()
        if not s:
            return None
        try:
            return pd.to_datetime(s, errors="coerce").date()
        except Exception:
            return None

    if ref.exists():
        try:
            df = pd.read_csv(ref, dtype=str).fillna("")
            for _, r in df.iterrows():
                c = r["canonical"].strip()
                if not c:
                    continue
                canon.add(c)
                al = [a.strip() for a in (r.get("aliases","").split("|")) if a.strip()]
                start = _d(r.get("start",""))
                end   = _d(r.get("end",""))
                rules.append({"canonical": c, "aliases": al, "start": start, "end": end})
                if al:
                    syns.setdefault(c, [])
                    for a in al:
                        if a not in syns[c]:
                            syns[c].append(a)
                mi = (r.get("merge_into","") or "").strip()
                ms = _d(r.get("merge_start",""))
                if mi:
                    merges[c] = (mi, ms)
        except Exception as e:
            print(f"[warn] could not load lineage file: {e}", flush=True)

    # Return canonical list (sorted), alias map, active map, and merge map
    return sorted(canon), syns, active, merges

def norm_camp_by_date(name: str, report_date: Optional[dt.date], canon_list, alias_map, merges):
    if not name:
        return name
    s = str(name).strip()
    low = s.lower()

    # 1) direct canonical hit
    if any(low == c.lower() for c in canon_list):
        canon = next(c for c in canon_list if c.lower() == low)
    else:
        # 2) alias → canonical (use time bounds if present in lineage)
        canon = s
        for c in canon_list:
            for a in alias_map.get(c, []):
                if low == a.strip().lower():
                    canon = c
                    break
            if canon == c:
                break

    # 3) apply merge if configured and start date reached
    tgt = merges.get(canon)
    if tgt:
        merge_into, merge_start = tgt
        if (report_date is None) or (merge_start is None) or (report_date >= merge_start):
            canon = merge_into

    return canon

def is_active(camp: str, report_date: dt.date, active_map) -> bool:
    if camp not in active_map:
        return True
    st, en = active_map[camp]
    if st and report_date < st:
        return False
    if en and report_date > en:
        return False
    return True

# ---------- Geometry parsing (pdfplumber) ----------
NUM_RE = re.compile(r"\d")  # token containing at least one digit

def _to_int_token(s: str):
    ds = re.sub(r"[^\d]", "", s or "")
    if not ds:
        return None
    try:
        return int(ds)
    except Exception:
        return None

HEADER_PATTERNS = {
    "tbbc_verified": re.compile(r"(?:\bTBC\b|\bTBBC\b|\bBBC\b).*verified", re.I),
    "tbbc_feeding":  re.compile(r"(?:\bTBC\b|\bTBBC\b|\bBBC\b).*(feeding|assisted)", re.I),
    "unhcr":         re.compile(r"(?:\bMOI/)?\bUNHCR\b.*(population|verified)?", re.I),
}

def _group_lines(words, y_tol=3):
    lines = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        placed = False
        for L in lines:
            if abs(L["top"] - w["top"]) <= y_tol:
                L["words"].append(w)
                L["top"] = min(L["top"], w["top"])
                L["bottom"] = max(L["bottom"], w["bottom"])
                placed = True
                break
        if not placed:
            lines.append({"top": w["top"], "bottom": w["bottom"], "words": [w]})
    for L in lines:
        L["x0"] = min(w["x0"] for w in L["words"])
        L["x1"] = max(w["x1"] for w in L["words"])
        L["text"] = " ".join(w["text"] for w in sorted(L["words"], key=lambda w: w["x0"]))
    return lines

def match_camp_from_text(txt: str, canon_ordered, syns) -> Optional[str]:
    t = normalize_whitespace(txt).lower()
    for c in canon_ordered:
        if re.search(r"\b" + re.escape(c.lower()) + r"\b", t):
            return c
        for a in syns.get(c, []):
            if re.search(r"\b" + re.escape(a.lower()) + r"\b", t):
                return c
    return None

def detect_columns(page) -> Dict[str, Tuple[float, float]]:
    # Find header bands by looking for TBBC/UNHCR labels and column titles near the top
    words = page.extract_words(use_text_flow=True, y_tolerance=3, x_tolerance=2, keep_blank_chars=False)
    lines = _group_lines(words)
    top = lines[:120]  # headers live near the top
    spans: Dict[str, Tuple[float, float]] = {}

    def union_span(match_fn):
        xs = [(L["x0"], L["x1"]) for L in top if match_fn(L["text"])]
        return (min(a for a, _ in xs), max(b for _, b in xs)) if xs else None

    rx_tbbc     = re.compile(r"\b(TBC|TBBC|T\.?B\.?C\.?|T\.?B\.?B\.?C\.?|BBC)\b", re.I)
    rx_verified = re.compile(r"\bverified(\s+caseload)?\b", re.I)
    rx_feeding  = re.compile(r"\b(feeding\s+figure|feeding|assisted|ff/?cf|cf/?ff|ff|cf)\b", re.I)
    rx_unhcr    = re.compile(r"\bUNHCR\b|\bMOI\b", re.I)
    rx_unh_aux  = re.compile(r"(population|verified)", re.I)

    # Direct column titles
    ver_span  = union_span(lambda t: rx_verified.search(t) and not rx_unhcr.search(t))
    feed_span = union_span(lambda t: rx_feeding.search(t))
    un_span   = union_span(lambda t: rx_unhcr.search(t) and rx_unh_aux.search(t))

    if ver_span:  spans["tbbc_verified"] = ver_span
    if feed_span: spans["tbbc_feeding"]  = feed_span
    if un_span:   spans["unhcr"]         = un_span

    # --- Fallback: anchor UNHCR by token if not detected above ---
    if "unhcr" not in spans:
        un_tokens = [w for w in words if re.search(r"\bUNHCR\b|\bMOI\b", w.get("text",""), re.I)]
        if un_tokens:
            x0_un = min(w["x0"] for w in un_tokens)
            x1_un = max(w["x1"] for w in un_tokens)
            page_right = max(w["x1"] for w in words)
            spans["unhcr"] = (x0_un - 5, page_right + 5)

    # If UNHCR exists, clamp TBBC spans strictly to its left so we never swallow UNHCR values
    if "unhcr" in spans:
        un_x0 = spans["unhcr"][0]
        for key in ("tbbc_verified", "tbbc_feeding"):
            if key in spans:
                x0, x1 = spans[key]
                spans[key] = (x0, min(x1, un_x0 - 6))

    return spans

def center_x(w):
    return (w["x0"] + w["x1"]) / 2.0

def center_y(w):
    return (w["top"] + w["bottom"]) / 2.0

def segment_rows(page, canon_ordered, syns, report_date: dt.date, active_map, y_pad=2):
    words = page.extract_words(use_text_flow=True, y_tolerance=3, x_tolerance=2, keep_blank_chars=False)
    lines = _group_lines(words)
    rows = []
    claimed = []  # y-span list
    for L in lines:
        camp = match_camp_from_text(L["text"], canon_ordered, syns)
        if not camp:
            continue
        if not is_active(camp, report_date, active_map):
            continue
        y0, y1 = L["top"] - y_pad, L["bottom"] + y_pad
        overlap = any(not (y1 < a or y0 > b) for (a, b) in claimed)
        if overlap:
            continue
        claimed.append((y0, y1))
        rows.append({"camp": camp, "y0": y0, "y1": y1, "x0": L["x0"], "x1": L["x1"]})
    rows.sort(key=lambda r: (r["y0"], r["camp"]))
    return rows

def extract_values_from_page(page, report_date: dt.date, source_url: str, cols: Dict[str, Tuple[float, float]], rows):
    words = page.extract_words(use_text_flow=True, y_tolerance=3, x_tolerance=2, keep_blank_chars=False)
    numerics = []
    for w in words:
        if NUM_RE.search(w.get("text","")) and _to_int_token(w.get("text","")) is not None:
            numerics.append(w)
    out = []
    for row in rows:
        y0, y1 = row["y0"], row["y1"]
        in_row = [w for w in numerics if y0 <= center_y(w) <= y1]
        for w in in_row:
            cx = center_x(w)
            series, subseries = None, ""
            val = _to_int_token(w.get("text",""))
            if val is None:
                continue
            if "tbbc_verified" in cols and cols["tbbc_verified"][0] <= cx <= cols["tbbc_verified"][1]:
                series, subseries = "tbbc", "verified"
            elif "tbbc_feeding" in cols and cols["tbbc_feeding"][0] <= cx <= cols["tbbc_feeding"][1]:
                series, subseries = "tbbc", "feeding"
            elif "unhcr" in cols and cols["unhcr"][0] <= cx <= cols["unhcr"][1]:
                series, subseries = "unhcr", "verified"
            else:
                continue
            out.append({
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
JOINER = r"[ _\-]*"
KNOWN_CAMPS_FALLBACK = [
    "Ban Mai Nai Soi","Ban Mae Surin","Mae La Oon","Mae Ra Ma Luang",
    "Mae La","Umpiem Mai","Nupo","Nu Po","Ban Don Yang","Tham Hin"
]
CAMP_PATS_FALLBACK = {
    camp: re.compile(fr"\b{JOINER.join(map(re.escape, camp.split()))}\b", re.I)
    for camp in KNOWN_CAMPS_FALLBACK
}

def find_header_and_columns(lines):
    pat_any_tbbc   = re.compile(r"(?:\bTBC\b|\bTBBC\b|\bBBC\b)", re.I)
    pat_verified   = re.compile(r"verified(\s+caseload)?", re.I)
    pat_feeding    = re.compile(r"(feeding\s+figure|feeding|assisted)", re.I)
    pat_unhcr      = re.compile(r"(?:\bMOI\b[^A-Za-z0-9/]*\/\s*)?\bUNHCR\b", re.I)
    pat_unhcr_aux  = re.compile(r"(population|verified)", re.I)

    N = min(80, len(lines))
    for i in range(N):
        win = lines[i:i+3]
        if not win:
            continue
        window_text = " ".join(win)
        if not ((pat_any_tbbc.search(window_text) and (pat_verified.search(window_text) or pat_feeding.search(window_text)))
                or (pat_unhcr.search(window_text) and pat_unhcr_aux.search(window_text))):
            continue

        # build rough column slices across this window
        pat_verified2 = re.compile(r"\bverified(\s+caseload)?\b", re.I)
        pat_feeding2  = re.compile(r"\b(feeding\s+figure|feeding|assisted)\b", re.I)
        pat_unhcr2    = re.compile(r"(?:\bMOI\b[^A-Za-z0-9/]*\/\s*)?\bUNHCR\b", re.I)
        pat_unh_aux2  = re.compile(r"(population|verified)", re.I)

        spans = []
        window_text = " ".join(win)
        m1 = pat_verified2.search(window_text)
        if m1:
            spans.append((m1.start(), m1.end(), "tbc_verified"))
        m2 = pat_feeding2.search(window_text)
        if m2:
            spans.append((m2.start(), m2.end(), "tbc_feeding"))
        m3 = pat_unhcr2.search(window_text)
        if m3 and pat_unh_aux2.search(window_text):
            spans.append((m3.start(), m3.end(), "unhcr"))

        if not spans:
            if pat_verified.search(window_text):
                spans.append((0, len(window_text), "tbc_tbbc_assist"))
            if pat_unhcr.search(window_text) and pat_unhcr_aux.search(window_text):
                spans.append((0, len(window_text), "unhcr"))

        if not spans:
            continue
        spans.sort(key=lambda x: x[0])

        expanded = []
        for j, (s, e, tag) in enumerate(spans):
            e2 = spans[j+1][0] if j+1 < len(spans) else len(window_text)
            expanded.append((s, e2, tag))

        cols, seen = [], set()
        for s, e, tag in expanded:
            ser = "unhcr" if tag == "unhcr" else "tbc"
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
                        for tok in re.findall(r"[0-9][0-9,\.]*\*?\+?\)?", seg):
                            val = _to_int_token(tok)
                            if val is None:
                                continue
                            found_vals.append((ser, val))
                else:
                    for tok in re.findall(r"[0-9][0-9,\.]*\*?\+?\)?", candidate_line):
                        val = _to_int_token(tok)
                        if val is None:
                            continue
                        found_vals.append(("tbc", val))
                if not found_vals:
                    continue
                for ser, val in found_vals:
                    if not (GLOBAL_MIN <= val <= GLOBAL_MAX):
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

# ---------- Normalization ----------
def _norm_series(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"tbc","tb bc","t.b.c.","t.b.b.c.","tbbc"}:
        return "tbbc"
    if s == "unhcr":
        return "unhcr"
    return s

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
        df_idx = df_idx.sort_values(["report_date", "file_name"], ascending=[False, True], na_position="last")

    if EXTRACT_MAX_FILES and EXTRACT_MAX_FILES > 0:
        df_idx = df_idx.head(EXTRACT_MAX_FILES)

    candidates = df_idx.to_dict(orient="records")
    canon, syns, active_map, merges = load_lineage()
  
    ensure_dirs()
    new_records = []

    for r in candidates:
        report_date = r["report_date"]
        url         = (r.get("source_url") or "").strip()
        fname       = (r.get("file_name")  or "").strip()
        rd_str      = report_date.strftime("%Y-%m-01") if pd.notna(report_date) else ""
        rd_date = None
        if rd_str:
            try:
                rd_date = dt.datetime.strptime(rd_str, "%Y-%m-%d").date()
            except Exception:
                rd_date = None
        if not url:
            continue

        local = download(url, rd_str)
        if local is None:
            continue

        ext = local.suffix.lower()

        # --- FAST PATH: geometry-first for PDFs; skip OCR unless explicitly enabled ---
        geom_rows: List[Dict] = []
        if ext == ".pdf":
            try:
                rd_date = dt.datetime.strptime(rd_str, "%Y-%m-%d").date() if rd_str else None
                if rd_date is not None:
                    geom_rows = parse_pdf_geometry(local, rd_date, url)
            except Exception as e:
                print(f"[warn] geometry parse failed on {local.name}: {e}", flush=True)

        if geom_rows:
            # If geometry found no UNHCR values, supplement via embedded-text fallback (no OCR)
            have_unhcr = any(rr.get("series") == "unhcr" for rr in geom_rows)
            if not have_unhcr:
                try:
                    import pdfplumber
                    parts = []
                    with pdfplumber.open(local) as pdf:
                        for p in pdf.pages:
                            t = p.extract_text(x_tolerance=2, y_tolerance=2) or ""
                            if t.strip():
                                parts.append(t)
                    embedded_text = "\n".join(parts)
                except Exception:
                    embedded_text = ""
                if embedded_text.strip():
                    extra = parse_rows_text(embedded_text)
                    have_keys = {(rr["camp"], rr["series"], rr.get("subseries","")) for rr in geom_rows}
                    for er in extra:
                        if er.get("series") == "unhcr" and (er["camp"], er["series"], er.get("subseries","")) not in have_keys:
                            geom_rows.append(er)

            # If geometry found no TBBC values, try embedded text supplement for TBBC as well
            have_tbbc = any(_norm_series(rr.get("series")) == "tbbc" for rr in geom_rows)
            if not have_tbbc:
                try:
                    import pdfplumber
                    parts2 = []
                    with pdfplumber.open(local) as pdf:
                        for p in pdf.pages:
                            t2 = p.extract_text(x_tolerance=3, y_tolerance=3) or ""
                            if t2.strip():
                                parts2.append(t2)
                    embedded_text2 = "\n".join(parts2)
                except Exception:
                    embedded_text2 = ""
                if (embedded_text2 or "").strip():
                    extra2 = parse_rows_text(embedded_text2)
                    have_keys2 = {(rr["camp"], _norm_series(rr["series"]), rr.get("subseries","")) for rr in geom_rows}
                    for er in extra2:
                        if _norm_series(er.get("series")) == "tbbc":
                            k2 = (er["camp"], "tbbc", er.get("subseries",""))
                            if k2 not in have_keys2:
                                geom_rows.append(er)

            category = "refugee"
            for row in geom_rows:
                new_records.append({
                    "report_date": rd_str,
                    "camp_name": norm_camp(row["camp"]),
                    "population": row["value"],
                    "category": category,
                    "series": _norm_series(row["series"]),
                    "subseries": row.get("subseries", ""),
                    "source_url": url,
                    "file_name": local.name,
                    "extract_method": "pdf-geometry",
                    "parse_notes": row.get("parse_notes", "geom_header_locked"),
                    "parse_confidence": row.get("parse_confidence", 0.98),
                })
            time.sleep(0.02)
            continue

        # --- FALLBACK: only now do text extraction; OCR can be disabled for speed ---
        text, method = "", None
        try:
            if ext in (".jpg", ".jpeg", ".png"):
                text, method = extract_text_from_image(local)
            elif ext == ".pdf":
                if DISABLE_OCR:
                    try:
                        import pdfplumber
                        with pdfplumber.open(local) as pdf:
                            parts = []
                            for p in pdf.pages:
                                t = p.extract_text(x_tolerance=2, y_tolerance=2) or ""
                                if t.strip():
                                    parts.append(t)
                            text = "\n".join(parts)
                        method = "pdf-text-no-ocr"
                    except Exception as e:
                        print(f"[warn] embedded text read failed (no OCR): {local.name} -> {e}", flush=True)
                        text, method = "", "pdf-text-no-ocr-fail"
                else:
                    text, method = extract_text_from_pdf(local)
            else:
                text, method = extract_text_from_image(local)
        except Exception as e:
            print(f"[warn] extraction failed: {local.name} -> {e}", flush=True)

        if not (text or "").strip():
            # emit a placeholder row to record that we touched this URL/date
            new_records.append({
                "report_date": rd_str, "camp_name": None, "population": None,
                "category": "refugee", "series": "unknown", "subseries": "",
                "source_url": url, "file_name": local.name,
                "extract_method": method or "unknown", "parse_notes": "no_text_extracted",
                "parse_confidence": 0.0
            })
            time.sleep(0.02)
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
                "camp_name": norm_camp(row.get("camp")),
                "population": row.get("value"),
                "category": detect_category(text or ""),
                "series": _norm_series(row.get("series", "unknown")),
                "subseries": row.get("subseries", ""),
                "source_url": url,
                "file_name": local.name,
                "extract_method": method or "unknown",
                "parse_notes": row.get("parse_notes", "text_line_locked"),
                "parse_confidence": row.get("parse_confidence", 0.9),
            })
        time.sleep(0.02)

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
        combined = (combined
                    .drop_duplicates(
                        subset=["report_date","camp_name","series","subseries","source_url","file_name","extract_method","parse_notes"],
                        keep="last"
                    )
                    .sort_values(["report_date","camp_name","series","subseries","file_name"], na_position="last"))
        # sanity filters
        with pd.option_context("mode.use_inf_as_na", True):
            combined["population"] = pd.to_numeric(combined["population"], errors="coerce")
        combined = combined[
            (combined["population"].isna()) |
            ((combined["population"] >= GLOBAL_MIN) & (combined["population"] <= GLOBAL_MAX))
        ]
        combined["camp_name"] = combined["camp_name"].map(norm_camp)
        combined.to_csv(OUT_LONG, index=False)
    else:
        pd.DataFrame().to_csv(OUT_LONG, index=False)

    # ---------- Write WIDE (preferred series per date) ----------
    try:
        if not combined.empty:
            dfw = combined.copy()
            with pd.option_context("mode.use_inf_as_na", True):
                dfw["report_date"] = pd.to_datetime(dfw["report_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            # pick preferred series per camp/date
            dfw = dfw.dropna(subset=["camp_name","report_date","series","population"])
            dfw = dfw[dfw["series"].str.lower().isin(["tbc","tbbc","unhcr"])]
            dfw["series_norm"] = dfw["series"].str.lower().replace({"tbc":"tbbc"})
            dfw = dfw.sort_values(["report_date","camp_name","series_norm"], key=lambda s: s.map({s:i for i,s in enumerate(SERIES_PREF)}))
            pref = dfw.drop_duplicates(subset=["report_date","camp_name"], keep="first")
            wide = (
                pref.pivot_table(
                    index="report_date",      # dates on index
                    columns="camp_name",      # camps as columns
                    values="population",
                    aggfunc="first"
                )
                .sort_index()
            )
            wide.to_csv(OUT_WIDE, index_label="report_date")
        else:
            pd.DataFrame().to_csv(OUT_WIDE, index=False)
    except Exception as e:
        print(f"[warn] could not generate WIDE CSV: {e}", flush=True)
        pd.DataFrame().to_csv(OUT_WIDE, index=False)

    # ---------- Write WIDE_DUAL (both series side-by-side) ----------
    try:
        if not combined.empty:
            dual = combined.copy()
            with pd.option_context("mode.use_inf_as_na", True):
                dual["report_date"] = pd.to_datetime(dual["report_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            dual = dual.dropna(subset=["camp_name","report_date","series","population"])
            dual["series"] = dual["series"].str.lower().replace({"tbc":"tbbc"})
            dual = dual[dual["series"].isin(["tbbc","unhcr"])]
            dual = dual[dual["population"].between(GLOBAL_MIN, GLOBAL_MAX)]
            dual["col"] = dual["report_date"] + "|" + dual["series"].str.lower()
            dual_w = dual.pivot_table(index="camp_name", columns="col", values="population", aggfunc="first")
            # Ensure both series columns exist per date for visibility
            all_dates = sorted(dual["report_date"].dropna().unique())
            want = []
            for d in all_dates:
                for s in ("tbbc","unhcr"):
                    want.append(f"{d}|{s}")
            for c in want:
                if c not in dual_w.columns:
                    dual_w[c] = pd.NA
            dual_w = dual_w[sorted(dual_w.columns)]
            dual_w.to_csv(OUT_WIDE_DUAL)
        else:
            pd.DataFrame().to_csv(OUT_WIDE_DUAL, index=False)
    except Exception as e:
        print(f"[warn] could not generate dual wide CSV: {e}", flush=True)
        pd.DataFrame().to_csv(OUT_WIDE_DUAL, index=False)

    print(f"[done] wrote {OUT_LONG} (+{len(df_new)} new rows, total {len(combined)})", flush=True)

# ---------- OCR helpers (only used when DISABLE_OCR=false) ----------
def extract_text_from_pdf(path: Path) -> Tuple[str,str]:
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text(x_tolerance=2, y_tolerance=2) or ""
                if t.strip():
                    parts.append(t)
        return "\n".join(parts), "pdf-text"
    except Exception as e:
        # fallback via OCR if needed
        return extract_text_from_image(path)

def extract_text_from_image(path: Path) -> Tuple[str,str]:
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(path)
        cfg = f"--psm {OCR_PSM}"
        txt = pytesseract.image_to_string(img, config=cfg)
        return txt or "", "image-ocr"
    except Exception as e:
        return "", "image-ocr-fail"

if __name__ == "__main__":
    main()
