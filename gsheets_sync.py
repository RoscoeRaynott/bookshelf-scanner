import os
import json
import re
import datetime
import streamlit as st


def get_canonical_key(title, author):
    clean_t = re.sub(r'[^a-zA-Z0-9]', '', str(title or "").lower())
    clean_a = re.sub(r'[^a-zA-Z0-9]', '', str(author or "").lower())
    return f"{clean_t}_{clean_a}"


try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    gspread = None
    Credentials = None
    GSHEETS_AVAILABLE = False


MASTER_CATALOG_HEADERS = [
    "Canonical Key", "ID", "Title", "Author", "Shelf",
    "Author Career Sales", "Book Sales / Listens", "TV / Film Deal",
    "Romance Rating", "Genre / Category", "Series / Protagonist",
    "Sightings", "Locations", "Search Evidence", "Last Updated"
]

BOOK_ARCHIVE_HEADERS = [
    "Canonical Key", "Title", "Author", "Book Sales",
    "TV Adaptation", "Sensual Rating", "Evidence", "Last Updated"
]

AUTHOR_ARCHIVE_HEADERS = [
    "Clean Author", "Author", "Author Career Sales", "Author Fame Score",
    "Evidence", "Last Updated"
]

LOCAL_MASTER_CATALOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "master_catalog.json")
LOCAL_GENRE_ARCHIVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "genre_archive.json")


def compute_author_fame_score(fame_str, raw_fallback=0.0):
    """Deterministically extract verified copy count, or strictly return 0.0 for unverified/midlist/missing."""
    if not fame_str:
        return 0.0
    f_str = str(fame_str).strip().lower()
    if f_str in ["-", "–", "n/a", "none", "unknown", ""]:
        return 0.0
    if any(term in f_str for term in ["not publicly", "midlist", "emerging", "unknown"]):
        return 0.0

    m_bil = re.search(r'(\d+(?:\.\d+)?)\s*billion\b', f_str)
    if m_bil:
        return float(m_bil.group(1)) * 1_000_000_000.0

    m_mil = re.search(r'(\d+(?:\.\d+)?)\s*(?:million\b|m\b)', f_str)
    if m_mil:
        return float(m_mil.group(1)) * 1_000_000.0

    m_k = re.search(r'(\d+(?:\.\d+)?)\s*k\b', f_str)
    if m_k:
        return float(m_k.group(1)) * 1_000.0

    m_raw = re.search(r'([\d,]{4,})\s*(?:copies|books|sales|sold)', f_str)
    if m_raw:
        digs = m_raw.group(1).replace(',', '')
        try:
            return float(digs)
        except Exception:
            pass

    return 0.0


def clean_cell(val):
    """Strip raw newlines and excessive whitespace so Google Sheets rows stay compact at single-line height."""
    if val is None:
        return "-"
    s = re.sub(r'\s+', ' ', str(val).replace('\r', ' ').replace('\n', ' ')).strip()
    return s if s else "-"


def is_gsheets_available():
    return GSHEETS_AVAILABLE



def get_service_account_dict():
    if not hasattr(st, "secrets"):
        return None
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
        if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
            return dict(st.secrets["connections"]["gsheets"])
    except Exception:
        pass
    return None


def get_spreadsheet_url():
    if not hasattr(st, "secrets"):
        return None
    try:
        if "gsheets" in st.secrets and "spreadsheet_url" in st.secrets["gsheets"]:
            return str(st.secrets["gsheets"]["spreadsheet_url"]).strip()
        if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
            conn = st.secrets["connections"]["gsheets"]
            return str(conn.get("spreadsheet") or conn.get("spreadsheet_url") or "").strip()
    except Exception:
        pass
    return None


def is_gsheets_configured():
    if not GSHEETS_AVAILABLE:
        return False
    sa = get_service_account_dict()
    url = get_spreadsheet_url()
    return bool(sa and sa.get("client_email") and url)


@st.cache_resource(show_spinner=False)
def get_gsheet_connection():
    if not GSHEETS_AVAILABLE:
        return None, "gspread or google-auth package is not installed."
    sa = get_service_account_dict()
    url = get_spreadsheet_url()
    if not sa or not url:
        return None, "Google Sheets credentials not found in st.secrets"

    if "private_key" in sa and isinstance(sa["private_key"], str):
        if "\\n" in sa["private_key"]:
            sa["private_key"] = sa["private_key"].replace("\\n", "\n")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    try:
        creds = Credentials.from_service_account_info(sa, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_url(url)
        return sh, None
    except Exception as ex:
        return None, str(ex)


def ensure_tab(sh, title, headers):
    if not GSHEETS_AVAILABLE or sh is None:
        return None
    try:
        ws = sh.worksheet(title)
    except Exception:
        try:
            ws = sh.add_worksheet(title=title, rows=1000, cols=len(headers) + 2)
            ws.append_row(headers)
            return ws
        except Exception:
            return None

    try:
        first_row = ws.row_values(1)
        if not first_row:
            ws.append_row(headers)
    except Exception:
        pass
    return ws


def load_all_from_gsheets(sh):
    books = []
    book_archive = {}
    author_archive = {}
    genre_archive = {}

    if sh is None:
        return books, book_archive, author_archive, genre_archive

    # 1. Master Catalog
    try:
        ws = ensure_tab(sh, "Master Catalog", MASTER_CATALOG_HEADERS)
        if ws:
            rows = ws.get_all_values()
            if len(rows) > 1:
                h_map = {str(name).strip(): idx for idx, name in enumerate(rows[0]) if str(name).strip()}
                for r in rows[1:]:
                    def _get(col_name, default=""):
                        idx = h_map.get(col_name)
                        if idx is not None and idx < len(r):
                            val = str(r[idx]).strip()
                            return val if val else default
                        return default

                    t = _get("Title")
                    if not t:
                        continue
                    k = _get("Canonical Key")
                    cat = _get("Genre / Category", "Standalone Novel")
                    ser_raw = _get("Series / Protagonist", "-")
                    if k and cat and cat != "-":
                        prot = "-"
                        ser_clean = ser_raw
                        if "(" in ser_raw and ser_raw.endswith(")"):
                            parts = ser_raw[:-1].split("(")
                            ser_clean = parts[0].strip()
                            prot = parts[1].strip()
                        genre_archive[k] = {
                            "category": cat,
                            "series": ser_clean or "Standalone Novel",
                            "protagonist": prot or "-"
                        }

                    books.append({
                        "id": int(_get("ID") or len(books) + 1),
                        "title": t,
                        "author": _get("Author"),
                        "shelf": int(_get("Shelf") or 1) if _get("Shelf").isdigit() else 1,
                        "author_fame": _get("Author Career Sales", "-"),
                        "sales": _get("Book Sales / Listens", "-"),
                        "tv_adaptation": _get("TV / Film Deal", "-"),
                        "sensual_romance_flag": _get("Romance Rating", "-"),
                        "category": cat or "Standalone Novel",
                        "series": ser_raw or "-",
                        "protagonist": "-",
                        "sightings_count": int(str(_get("Sightings", "1")).replace("x", "") or 1) if str(_get("Sightings", "1")).replace("x", "").isdigit() else 1,
                        "all_locations": [loc.strip() for loc in _get("Locations", "").split(",") if loc.strip()],
                        "search_evidence": _get("Search Evidence", "-"),
                        "deep_searched": bool(_get("Book Sales / Listens") and _get("Book Sales / Listens") != "-"),
                        "source": "Google Sheets"
                    })
    except Exception as e:
        print(f"Error loading Master Catalog: {e}")

    # 2. Book Archive
    try:
        ws_b = ensure_tab(sh, "Book Search Archive", BOOK_ARCHIVE_HEADERS)
        if ws_b:
            b_rows = ws_b.get_all_values()
            if len(b_rows) > 1:
                b_map = {str(name).strip(): idx for idx, name in enumerate(b_rows[0]) if str(name).strip()}
                for r in b_rows[1:]:
                    def _b_get(col_name, default=""):
                        idx = b_map.get(col_name)
                        if idx is not None and idx < len(r):
                            val = str(r[idx]).strip()
                            return val if val else default
                        return default

                    k = _b_get("Canonical Key")
                    if k:
                        book_archive[k] = {
                            "book_sales": _b_get("Book Sales", "-"),
                            "tv_adaptation": _b_get("TV Adaptation", "-"),
                            "sensual_rating": _b_get("Sensual Rating", "-"),
                            "evidence": _b_get("Evidence", "-")
                        }
    except Exception as e:
        print(f"Error loading Book Archive: {e}")

    # 3. Author Archive
    try:
        ws_a = ensure_tab(sh, "Author Archive", AUTHOR_ARCHIVE_HEADERS)
        if ws_a:
            a_rows = ws_a.get_all_values()
            if len(a_rows) > 1:
                a_map = {str(name).strip(): idx for idx, name in enumerate(a_rows[0]) if str(name).strip()}
                for r in a_rows[1:]:
                    def _a_get(col_name, default=""):
                        idx = a_map.get(col_name)
                        if idx is not None and idx < len(r):
                            val = str(r[idx]).strip()
                            return val if val else default
                        return default

                    k = _a_get("Clean Author").lower()
                    if k:
                        fame = _a_get("Author Career Sales", "-")
                        fame_score = compute_author_fame_score(fame, _a_get("Author Fame Score", "0"))
                        author_archive[k] = {
                            "author": _a_get("Author", k),
                            "author_fame": fame,
                            "author_fame_score": fame_score,
                            "evidence": _a_get("Evidence", "-")
                        }
    except Exception as e:
        print(f"Error loading Author Archive: {e}")

    return books, book_archive, author_archive, genre_archive


def sync_catalog_to_gsheets(sh, books, get_canonical_key_fn=None):
    if not sh:
        return False, "Google Sheet connection not available"
    if not books:
        return False, "Catalog is empty (no books detected yet)"
    try:
        ws = ensure_tab(sh, "Master Catalog", MASTER_CATALOG_HEADERS)
        if not ws:
            return False, "Could not open or create 'Master Catalog' tab"

        existing_data = ws.get_all_values()
        row_dict = {}
        if len(existing_data) > 1:
            for r in existing_data[1:]:
                if r and len(r) > 0 and r[0].strip():
                    padded = r + ["-"] * max(0, 15 - len(r))
                    row_dict[r[0].strip()] = padded[:15]

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for b in books:
            t = str(b.get("title") or "").strip()
            a = str(b.get("author") or "").strip()
            if not t or "unidentified" in t.lower() or t.lower().startswith("book "):
                continue
            fn = get_canonical_key_fn or get_canonical_key
            c_key = fn(t, a)

            ser_str = f"{b.get('series')} ({b.get('protagonist')})" if b.get("protagonist") and b.get("protagonist") != "-" else str(b.get("series") or "-")
            loc_str = ", ".join(b.get("all_locations", [])) if b.get("all_locations") else f"Shelf {b.get('shelf', 1)}"

            if c_key in row_dict:
                existing = row_dict[c_key]
                new_sales = str(b.get("sales") or "-")
                if new_sales != "-" and existing[6] == "-":
                    existing[6] = new_sales
                new_tv = str(b.get("tv_adaptation") or "-")
                if new_tv != "-" and existing[7] == "-":
                    existing[7] = new_tv
                new_romance = str(b.get("sensual_romance_flag") or "-")
                if new_romance != "-" and existing[8] == "-":
                    existing[8] = new_romance
                new_cat = str(b.get("category") or "-")
                if new_cat not in ("-", "Standalone Novel") and existing[9] in ("-", "Standalone Novel"):
                    existing[9] = new_cat
                if ser_str != "-" and existing[10] == "-":
                    existing[10] = ser_str
                existing[11] = f"{b.get('sightings_count', 1)}x"
                if loc_str and loc_str not in existing[12]:
                    existing[12] = f"{existing[12]}, {loc_str}" if existing[12] and existing[12] != "-" else loc_str
                existing[14] = now_str
            else:
                row_vals = [
                    c_key,
                    str(b.get("id") or len(row_dict) + 1),
                    t,
                    a,
                    str(b.get("shelf", 1)),
                    str(b.get("author_fame") or "-"),
                    str(b.get("sales") or "-"),
                    str(b.get("tv_adaptation") or "-"),
                    str(b.get("sensual_romance_flag") or "-"),
                    str(b.get("category") or "Standalone Novel"),
                    ser_str,
                    f"{b.get('sightings_count', 1)}x",
                    loc_str,
                    str(b.get("search_evidence") or "-"),
                    now_str
                ]
                row_dict[c_key] = row_vals

        if not row_dict:
            return False, "No valid book titles found to sync"

        all_table = [MASTER_CATALOG_HEADERS] + list(row_dict.values())
        clean_table = [[clean_cell(c) for c in row] for row in all_table]
        if ws.row_count < len(clean_table):
            ws.add_rows(len(clean_table) - ws.row_count + 20)
        ws.clear()
        ws.update(clean_table, "A1", raw=False)
        return True, f"Synced {len(row_dict)} books to Google Sheets"
    except Exception as ex:
        print(f"Error syncing catalog to Google Sheets: {ex}")
        return False, str(ex)


def sync_authors_to_gsheets(sh, author_archive):
    if not sh or not author_archive:
        return True, "No authors to sync"
    try:
        ws_a = ensure_tab(sh, "Author Archive", AUTHOR_ARCHIVE_HEADERS)
        if not ws_a:
            return False, "Could not access Author Archive tab"

        existing_data = ws_a.get_all_values()
        row_dict = {}
        if len(existing_data) > 1:
            for r in existing_data[1:]:
                if r and len(r) > 0 and r[0].strip():
                    padded = r + ["-"] * max(0, 6 - len(r))
                    clean_k = r[0].strip().lower()
                    fame_txt = padded[2]
                    score_val = compute_author_fame_score(fame_txt, padded[3])
                    padded[3] = str(int(score_val) if score_val.is_integer() else score_val)
                    row_dict[clean_k] = padded[:6]

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for clean_author, data in author_archive.items():
            k = clean_author.strip().lower()
            a_name = str(data.get("author") or clean_author)
            fame = str(data.get("author_fame") or "-")
            score_val = compute_author_fame_score(fame, data.get("author_fame_score"))
            score = str(int(score_val) if score_val.is_integer() else score_val)
            ev = str(data.get("evidence") or "-")
            row_dict[k] = [k, a_name, fame, score, ev, now_str]

        all_table = [AUTHOR_ARCHIVE_HEADERS] + list(row_dict.values())
        clean_table = [[clean_cell(c) for c in row] for row in all_table]
        if ws_a.row_count < len(clean_table):
            ws_a.add_rows(len(clean_table) - ws_a.row_count + 20)
        ws_a.clear()
        ws_a.update(clean_table, "A1", raw=False)
        return True, f"Synced {len(row_dict)} authors"
    except Exception as ex:
        print(f"Error syncing authors to Google Sheets: {ex}")
        return False, str(ex)


def sync_books_to_gsheets(sh, book_archive):
    if not sh or not book_archive:
        return True, "No book search cache to sync"
    try:
        ws_b = ensure_tab(sh, "Book Search Archive", BOOK_ARCHIVE_HEADERS)
        if not ws_b:
            return False, "Could not access Book Search Archive tab"

        existing_data = ws_b.get_all_values()
        row_dict = {}
        if len(existing_data) > 1:
            for r in existing_data[1:]:
                if r and len(r) > 0 and r[0].strip():
                    padded = r + ["-"] * max(0, 8 - len(r))
                    row_dict[r[0].strip()] = padded[:8]

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for canon_key, data in book_archive.items():
            k = canon_key.strip()
            row_dict[k] = [
                k,
                str(data.get("title") or k),
                str(data.get("author") or "-"),
                str(data.get("book_sales") or "-"),
                str(data.get("tv_adaptation") or "-"),
                str(data.get("sensual_rating") or "-"),
                str(data.get("evidence") or "-"),
                now_str
            ]

        all_table = [BOOK_ARCHIVE_HEADERS] + list(row_dict.values())
        clean_table = [[clean_cell(c) for c in row] for row in all_table]
        if ws_b.row_count < len(clean_table):
            ws_b.add_rows(len(clean_table) - ws_b.row_count + 20)
        ws_b.clear()
        ws_b.update(clean_table, "A1", raw=False)
        return True, f"Synced {len(row_dict)} book search records"
    except Exception as ex:
        print(f"Error syncing books to Google Sheets: {ex}")
        return False, str(ex)



def sync_book_search_to_gsheets(sh, canon_key, title, author, res):
    if not sh:
        return
    try:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        ws_b = ensure_tab(sh, "Book Search Archive", BOOK_ARCHIVE_HEADERS)
        if ws_b:
            b_vals = [
                clean_cell(canon_key),
                clean_cell(title),
                clean_cell(author),
                clean_cell(res.get("book_sales")),
                clean_cell(res.get("tv_adaptation")),
                clean_cell(res.get("sensual_rating")),
                clean_cell(res.get("evidence")),
                now_str
            ]
            existing = ws_b.get_all_values()
            found_idx = None
            if len(existing) > 1:
                for idx, r in enumerate(existing[1:], start=2):
                    if r and r[0].strip() == canon_key:
                        found_idx = idx
                        break
            if found_idx:
                ws_b.update(values=[b_vals], range_name=f"A{found_idx}:H{found_idx}")
            else:
                ws_b.append_row(b_vals, value_input_option="USER_ENTERED")

        ws_m = ensure_tab(sh, "Master Catalog", MASTER_CATALOG_HEADERS)
        if ws_m:
            m_data = ws_m.get_all_values()
            if len(m_data) > 1:
                for idx, r in enumerate(m_data[1:], start=2):
                    if r and r[0].strip() == canon_key:
                        ws_m.update_cell(idx, 7, clean_cell(res.get("book_sales")))
                        ws_m.update_cell(idx, 8, clean_cell(res.get("tv_adaptation")))
                        ws_m.update_cell(idx, 9, clean_cell(res.get("sensual_rating")))
                        ws_m.update_cell(idx, 14, clean_cell(res.get("evidence")))
                        ws_m.update_cell(idx, 15, now_str)
                        break
    except Exception as ex:
        print(f"Error syncing book search to Google Sheets: {ex}")


def sync_author_to_gsheets(sh, clean_author, author_name, fame, fame_score, evidence):
    if not sh:
        return
    try:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        ws_a = ensure_tab(sh, "Author Archive", AUTHOR_ARCHIVE_HEADERS)
        if ws_a:
            score_val = compute_author_fame_score(fame, fame_score)
            score_str = str(int(score_val) if score_val.is_integer() else score_val)
            a_vals = [
                clean_cell(clean_author),
                clean_cell(author_name),
                clean_cell(fame),
                score_str,
                clean_cell(evidence),
                now_str
            ]
            existing = ws_a.get_all_values()
            found_idx = None
            if len(existing) > 1:
                for idx, r in enumerate(existing[1:], start=2):
                    if r and r[0].strip().lower() == clean_author.lower():
                        found_idx = idx
                        break
            if found_idx:
                ws_a.update(values=[a_vals], range_name=f"A{found_idx}:F{found_idx}")
            else:
                ws_a.append_row(a_vals, value_input_option="USER_ENTERED")
    except Exception as ex:
        print(f"Error syncing author to Google Sheets: {ex}")


def load_local_master_catalog():
    if os.path.exists(LOCAL_MASTER_CATALOG_FILE):
        try:
            with open(LOCAL_MASTER_CATALOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_local_master_catalog(books):
    try:
        os.makedirs(os.path.dirname(LOCAL_MASTER_CATALOG_FILE), exist_ok=True)
        clean_list = []
        for b in books:
            clean_list.append({
                "id": b.get("id"),
                "title": b.get("title"),
                "author": b.get("author"),
                "shelf": b.get("shelf", 1),
                "author_fame": b.get("author_fame", "-"),
                "author_fame_score": b.get("author_fame_score", 0.0),
                "sales": b.get("sales", "-"),
                "tv_adaptation": b.get("tv_adaptation", "-"),
                "sensual_romance_flag": b.get("sensual_romance_flag", "-"),
                "category": b.get("category", "Standalone Novel"),
                "series": b.get("series", "-"),
                "protagonist": b.get("protagonist", "-"),
                "sightings_count": b.get("sightings_count", 1),
                "all_locations": b.get("all_locations", []),
                "search_evidence": b.get("search_evidence", "-"),
                "deep_searched": bool(b.get("deep_searched", False)),
                "source": b.get("source", "API")
            })
        with open(LOCAL_MASTER_CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_list, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_local_genre_archive():
    if os.path.exists(LOCAL_GENRE_ARCHIVE_FILE):
        try:
            with open(LOCAL_GENRE_ARCHIVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_local_genre_archive(archive):
    try:
        os.makedirs(os.path.dirname(LOCAL_GENRE_ARCHIVE_FILE), exist_ok=True)
        with open(LOCAL_GENRE_ARCHIVE_FILE, "w", encoding="utf-8") as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

