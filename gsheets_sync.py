import os
import json
import datetime
import streamlit as st

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
    "Canonical Key", "Title", "Author", "Book Sales", "Sales Score",
    "TV Adaptation", "Sensual Rating", "Evidence", "Last Updated"
]

AUTHOR_ARCHIVE_HEADERS = [
    "Clean Author", "Author", "Author Career Sales", "Author Fame Score",
    "Evidence", "Last Updated"
]

LOCAL_MASTER_CATALOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "master_catalog.json")
LOCAL_GENRE_ARCHIVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "genre_archive.json")


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
            records = ws.get_all_records()
            for r in records:
                if not r.get("Title"):
                    continue
                k = str(r.get("Canonical Key") or "").strip()
                cat = str(r.get("Genre / Category") or "").strip()
                ser_raw = str(r.get("Series / Protagonist") or "").strip()
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
                    "id": int(r.get("ID") or len(books) + 1),
                    "title": str(r.get("Title") or "").strip(),
                    "author": str(r.get("Author") or "").strip(),
                    "shelf": int(r.get("Shelf") or 1),
                    "author_fame": str(r.get("Author Career Sales") or "-"),
                    "sales": str(r.get("Book Sales / Listens") or "-"),
                    "sales_score": 10.0 if "bestseller" in str(r.get("Book Sales / Listens") or "").lower() or "million" in str(r.get("Book Sales / Listens") or "").lower() else 0.0,
                    "tv_adaptation": str(r.get("TV / Film Deal") or "-"),
                    "sensual_romance_flag": str(r.get("Romance Rating") or "-"),
                    "category": cat or "Standalone Novel",
                    "series": ser_raw or "-",
                    "protagonist": "-",
                    "sightings_count": int(str(r.get("Sightings") or "1").replace("x", "") or 1),
                    "all_locations": [loc.strip() for loc in str(r.get("Locations") or "").split(",") if loc.strip()],
                    "search_evidence": str(r.get("Search Evidence") or "-"),
                    "deep_searched": bool(r.get("Book Sales / Listens") and str(r.get("Book Sales / Listens")) != "-"),
                    "source": "Google Sheets"
                })
    except Exception as e:
        print(f"Error loading Master Catalog: {e}")

    # 2. Book Archive
    try:
        ws_b = ensure_tab(sh, "Book Search Archive", BOOK_ARCHIVE_HEADERS)
        if ws_b:
            b_records = ws_b.get_all_records()
            for r in b_records:
                k = str(r.get("Canonical Key") or "").strip()
                if k:
                    book_archive[k] = {
                        "book_sales": str(r.get("Book Sales") or "-"),
                        "sales_score": float(r.get("Sales Score") or 0.0),
                        "tv_adaptation": str(r.get("TV Adaptation") or "-"),
                        "sensual_rating": str(r.get("Sensual Rating") or "-"),
                        "evidence": str(r.get("Evidence") or "-")
                    }
    except Exception as e:
        print(f"Error loading Book Archive: {e}")

    # 3. Author Archive
    try:
        ws_a = ensure_tab(sh, "Author Archive", AUTHOR_ARCHIVE_HEADERS)
        if ws_a:
            a_records = ws_a.get_all_records()
            for r in a_records:
                k = str(r.get("Clean Author") or "").strip().lower()
                if k:
                    author_archive[k] = {
                        "author": str(r.get("Author") or ""),
                        "author_fame": str(r.get("Author Career Sales") or "-"),
                        "author_fame_score": float(r.get("Author Fame Score") or 0.0),
                        "evidence": str(r.get("Evidence") or "-")
                    }
    except Exception as e:
        print(f"Error loading Author Archive: {e}")

    return books, book_archive, author_archive, genre_archive


def sync_catalog_to_gsheets(sh, books, get_canonical_key_fn):
    if not sh or not books:
        return
    try:
        ws = ensure_tab(sh, "Master Catalog", MASTER_CATALOG_HEADERS)
        if not ws:
            return

        existing_data = ws.get_all_values()
        existing_keys = {}
        if len(existing_data) > 1:
            for row_idx, row in enumerate(existing_data[1:], start=2):
                if row and len(row) > 0:
                    k = row[0].strip()
                    if k:
                        existing_keys[k] = row_idx

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        to_append = []

        for b in books:
            t = str(b.get("title") or "").strip()
            a = str(b.get("author") or "").strip()
            if not t or "unidentified" in t.lower() or t.lower().startswith("book "):
                continue
            c_key = get_canonical_key_fn(t, a)

            row_vals = [
                c_key,
                str(b.get("id") or ""),
                t,
                a,
                str(b.get("shelf", 1)),
                str(b.get("author_fame") or "-"),
                str(b.get("sales") or "-"),
                str(b.get("tv_adaptation") or "-"),
                str(b.get("sensual_romance_flag") or "-"),
                str(b.get("category") or "Standalone Novel"),
                f"{b.get('series')} ({b.get('protagonist')})" if b.get("protagonist") and b.get("protagonist") != "-" else str(b.get("series") or "-"),
                f"{b.get('sightings_count', 1)}x",
                ", ".join(b.get("all_locations", [])),
                str(b.get("search_evidence") or "-"),
                now_str
            ]

            if c_key in existing_keys:
                row_idx = existing_keys[c_key]
                ws.update(values=[row_vals], range_name=f"A{row_idx}:O{row_idx}")
            else:
                to_append.append(row_vals)
                existing_keys[c_key] = len(existing_data) + len(to_append)

        if to_append:
            ws.append_rows(to_append, value_input_option="USER_ENTERED")
    except Exception as ex:
        print(f"Error syncing catalog to Google Sheets: {ex}")


def sync_book_search_to_gsheets(sh, canon_key, title, author, res):
    if not sh:
        return
    try:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        ws_b = ensure_tab(sh, "Book Search Archive", BOOK_ARCHIVE_HEADERS)
        if ws_b:
            b_vals = [
                canon_key,
                title,
                author,
                str(res.get("book_sales") or "-"),
                str(res.get("sales_score") or 0),
                str(res.get("tv_adaptation") or "-"),
                str(res.get("sensual_rating") or "-"),
                str(res.get("evidence") or "-"),
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
                ws_b.update(values=[b_vals], range_name=f"A{found_idx}:I{found_idx}")
            else:
                ws_b.append_row(b_vals, value_input_option="USER_ENTERED")

        ws_m = ensure_tab(sh, "Master Catalog", MASTER_CATALOG_HEADERS)
        if ws_m:
            m_data = ws_m.get_all_values()
            if len(m_data) > 1:
                for idx, r in enumerate(m_data[1:], start=2):
                    if r and r[0].strip() == canon_key:
                        ws_m.update_cell(idx, 7, str(res.get("book_sales") or "-"))
                        ws_m.update_cell(idx, 8, str(res.get("tv_adaptation") or "-"))
                        ws_m.update_cell(idx, 9, str(res.get("sensual_rating") or "-"))
                        ws_m.update_cell(idx, 14, str(res.get("evidence") or "-"))
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
            a_vals = [
                clean_author,
                author_name,
                str(fame or "-"),
                str(fame_score or 0),
                str(evidence or "-"),
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
                "sales_score": b.get("sales_score", 0.0),
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

