import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import json
import os
import re
import base64
import io
import time
import urllib.request
import urllib.error
import urllib.parse
from PIL import Image, ImageOps
import concurrent.futures
import math
from gsheets_sync import (
    is_gsheets_configured,
    get_gsheet_connection,
    load_all_from_gsheets,
    sync_catalog_to_gsheets,
    sync_book_search_to_gsheets,
    sync_author_to_gsheets,
    sync_authors_to_gsheets,
    sync_books_to_gsheets,
    load_local_master_catalog,
    save_local_master_catalog,
    load_local_genre_archive,
    save_local_genre_archive,
    is_gsheets_available,
    get_canonical_key,
    compute_author_fame_score,
)
import importlib
import arena_benchmark
try:
    importlib.reload(arena_benchmark)
except Exception:
    pass
from arena_benchmark import (
    query_direct_gemini_api,
    query_direct_gemini_author_fame,
    run_vision_benchmark_openrouter,
    run_vision_benchmark_direct_gemini,
    draw_annotated_vision_result,
    _execute_with_rate_limit_retry,
    _extract_text_from_resp,
    _parse_json_result,
)

_CLIENT_UPLOADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_uploader")
if os.path.exists(_CLIENT_UPLOADER_DIR):
    _client_uploader = components.declare_component("fast_shelf_uploader", path=_CLIENT_UPLOADER_DIR)
else:
    _client_uploader = None

_SHELF_INSPECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shelf_inspector")
if os.path.exists(_SHELF_INSPECTOR_DIR):
    _shelf_inspector = components.declare_component("shelf_inspector", path=_SHELF_INSPECTOR_DIR)
else:
    _shelf_inspector = None

_SHELF_PINPOINTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shelf_pinpointer")
if os.path.exists(_SHELF_PINPOINTER_DIR):
    _shelf_pinpointer = components.declare_component("shelf_pinpointer", path=_SHELF_PINPOINTER_DIR)
else:
    _shelf_pinpointer = None

# Pixel phones in "High efficiency" mode hand the browser a .heic file, which
# neither cv2 nor stock Pillow can decode. Without this the photo uploads fine
# and then silently decodes to nothing.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except Exception:
    HEIF_SUPPORTED = False

st.set_page_config(
    page_title="Multi-Shelf Book Scanner & Cataloger",
    page_icon="📚",
    layout="wide",
    # "auto" keeps the sidebar open on a laptop but collapses it on a phone,
    # where "expanded" covers the entire screen including the upload button.
    initial_sidebar_state="auto"
)

st.title("📚 Multi-Bookshelf Scanner & Master Cataloger")
st.caption("AI-Powered Book Scanner: Fast vision models to detect, segment, and catalog every book on multi-column shelves.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGE = os.path.join(BASE_DIR, "data", "sample_shelf.jpg")
ANNOTATED_IMAGE = os.path.join(BASE_DIR, "data", "annotated_bookshelf_rotated.jpg")

AUTHOR_ARCHIVE_FILE = os.path.join(BASE_DIR, "data", "author_archive.json")
BOOK_ARCHIVE_FILE = os.path.join(BASE_DIR, "data", "book_archive.json")
GENRE_ARCHIVE_FILE = os.path.join(BASE_DIR, "data", "genre_archive.json")


def load_author_archive():
    if os.path.exists(AUTHOR_ARCHIVE_FILE):
        try:
            with open(AUTHOR_ARCHIVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_author_archive(archive):
    try:
        os.makedirs(os.path.dirname(AUTHOR_ARCHIVE_FILE), exist_ok=True)
        with open(AUTHOR_ARCHIVE_FILE, "w", encoding="utf-8") as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_book_archive():
    if os.path.exists(BOOK_ARCHIVE_FILE):
        try:
            with open(BOOK_ARCHIVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_book_archive(archive):
    try:
        os.makedirs(os.path.dirname(BOOK_ARCHIVE_FILE), exist_ok=True)
        with open(BOOK_ARCHIVE_FILE, "w", encoding="utf-8") as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_genre_archive():
    if os.path.exists(GENRE_ARCHIVE_FILE):
        try:
            with open(GENRE_ARCHIVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_genre_archive(archive):
    try:
        os.makedirs(os.path.dirname(GENRE_ARCHIVE_FILE), exist_ok=True)
        with open(GENRE_ARCHIVE_FILE, "w", encoding="utf-8") as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_UPLOAD_DIM = 8192       # 8K resolution: full 100% native camera sensor resolution (up to 50MP)
JPEG_QUALITY = 95           # Maximum visual sharpness for spine OCR
MAX_OUTPUT_TOKENS = 16384   # Accommodate 100+ book shelves without cutoff
STREAM_STALL_TIMEOUT = 90   # seconds of total silence from the server before giving up
HARD_DEADLINE = 240         # seconds for one photo, across all retries of a single call
MAX_RETRIES = 2
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "heic", "heif"]

# Session state is initialised before any widget so the sidebar Reset button can
# clear it on the same run it is pressed.
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}
if "master_books" not in st.session_state:
    st.session_state.master_books = []
if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = {}
if "uploader_nonce" not in st.session_state:
    st.session_state.uploader_nonce = 0
if "last_client_batch_id" not in st.session_state:
    st.session_state.last_client_batch_id = ""
if "use_fallback_uploader" not in st.session_state:
    st.session_state.use_fallback_uploader = False
if "custom_shelf_dividers" not in st.session_state:
    st.session_state.custom_shelf_dividers = {}

# ---------------------------------------------------------------------------
# OpenRouter vision API
#
# Everything here streams. A shelf photo can legitimately take 30-90s to
# describe -- a single blocking request gives the UI nothing to show for that
# whole time, which is indistinguishable from a hang. Streaming lets the caller
# report bytes-arriving and seconds-elapsed while the model writes.
# ---------------------------------------------------------------------------

VISION_PROMPT = """Analyze this bookstore bookshelf image. Detect and catalog every book visible across all shelves from top to bottom.
Scan thoroughly shelf by shelf, from top to bottom, and on each shelf strictly from left to right.
Be sure to detect all books on the bottom-most shelf near the bottom edge of the frame.

CRITICAL GROUNDING & ANTI-HALLUCINATION RULES:
1. STRICT SPINE-ONLY TRANSCRIPTION: For each book, you MUST transcribe ONLY the exact characters visible on that specific physical spine ("spine_text").
   - NEVER substitute another title from that author's bibliography! If an author is known (e.g. James Patterson, Lee Child, Nora Roberts), do NOT guess a popular title from memory if it is not written on this spine.
   - The "title" and "author" MUST strictly correspond to the letters visible on that spine.
2. SERIES NUMBERS: If an explicit book number (e.g., "#1", "Book 2", "Vol 3") is printed on the spine, include it. If no number is printed on the spine, DO NOT guess or hallucinate a series number.
3. BLURRY / UNREADABLE SPINES: If a spine is too dark, narrow, or blurry to read clearly, set "spine_text": "Unreadable", "title": "Unidentified Book", "author": "Unknown". It is 100x better to report "Unidentified Book" than to guess a real book from the author's catalog that is NOT on the shelf.
4. DUPLICATE COPIES: Bookstores frequently shelve 2 or more identical copies of the same book side-by-side. You MUST create a separate entry for EVERY physical spine with its own bounding box. NEVER collapse or skip duplicate copies.
5. ORDER & TILT: Order entries shelf by shelf, and left to right (increasing xmin). Estimate "tilt_angle" in degrees from vertical (-30 to +30, 0 = upright, negative = leaning left, positive = leaning right).
6. MAIN SHELF ONLY: Catalog ONLY books standing upright on the primary shelf of this image. Completely ignore cut-off book tops, bottoms, or partial slivers peeking in across the top or bottom frame borders.

Return a valid JSON object:
{
  "books": [
    {
      "box_2d": [ymin, xmin, ymax, xmax],
      "tilt_angle": 0,
      "shelf_row": 1,
      "spine_text": "Exact text visible on this spine",
      "title": "Canonical Title",
      "author": "Author Name"
    }
  ]
}
Coordinates: "box_2d" normalized integers 0 to 1000 representing [ymin, xmin, ymax, xmax]."""


def _api_headers(key):
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://bookshelf-scanner.streamlit.app",
        "X-Title": "Bookshelf Scanner",
    }


def _http_error_message(err):
    """OpenRouter puts the useful reason in the response body, not the status line."""
    try:
        body = json.loads(err.read().decode("utf-8", "ignore"))
        detail = body.get("error", {})
        if isinstance(detail, dict) and detail.get("message"):
            return f"HTTP {err.code}: {detail['message']}"
        return f"HTTP {err.code}: {body}"
    except Exception:
        return f"HTTP {err.code}: {err.reason}"


def _extract_json(text):
    """Parse a JSON object out of a reply that may be fenced, padded, or truncated."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    # Pass 1: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Pass 2: find matching root braces
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except Exception:
            pass

    # Pass 3: salvage truncated array if model hit output token ceiling
    if first != -1 and ('"books"' in text or "'books'" in text):
        last_obj = text.rfind("}")
        if last_obj > first:
            repaired = text[first:last_obj + 1] + "\n  ]\n}"
            try:
                data = json.loads(repaired)
                if isinstance(data, dict) and "books" in data:
                    return data
            except Exception:
                pass

    return None


def _stream_completion(payload, key, status_cb, started, label):
    """POST with stream=True and assemble the reply, reporting progress as it lands."""
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=_api_headers(key),
    )
    chunks = []
    n_chars = 0
    # Negative seed so the very first content chunk always reports: the switch
    # from "queued" to "writing" is the moment that proves it is not hung.
    last_tick = -1.0

    with urllib.request.urlopen(req, timeout=STREAM_STALL_TIMEOUT) as resp:
        for raw in resp:
            elapsed = time.time() - started
            if elapsed > HARD_DEADLINE:
                raise TimeoutError(
                    f"Gave up after {HARD_DEADLINE}s. The model was still writing "
                    f"({n_chars} characters so far) -- try a smaller photo or Flash-Lite."
                )
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            if line.startswith(":"):
                # ": OPENROUTER PROCESSING" keepalives while the request is queued.
                status_cb(f"⏳ {label}: queued at OpenRouter… {elapsed:.0f}s")
                continue
            if not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except Exception:
                continue
            if event.get("error"):
                detail = event["error"]
                raise RuntimeError(
                    detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                )
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                reasoning_chunk = delta.get("reasoning") or delta.get("thought") or ""
                piece = delta.get("content") or ""
                if reasoning_chunk and elapsed - last_tick > 0.4:
                    last_tick = elapsed
                    status_cb(f"🧠 {label}: reasoning & analyzing spine text… {elapsed:.0f}s")
                if not piece:
                    continue
                chunks.append(piece)
                n_chars += len(piece)
                # Throttle: one status write every 0.4s, not one per token.
                if elapsed - last_tick > 0.4:
                    last_tick = elapsed
                    status_cb(
                        f"✍️ {label}: model is writing… "
                        f"{n_chars} chars · {elapsed:.0f}s elapsed"
                    )
    return "".join(chunks)


def call_vision_api(img_bgr, model_id, key, status_cb=None):
    """Send one shelf photo to the model and return the parsed list of books."""
    status_cb = status_cb or (lambda _msg: None)
    short_name = model_id.split("/")[-1]

    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not success:
        st.error("Could not JPEG-encode this photo before sending it.")
        return []
    if len(buffer) > 14 * 1024 * 1024:
        success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_img = base64.b64encode(buffer).decode("utf-8")
    kb = len(buffer) // 1024

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        started = time.time()
        suffix = "" if attempt == 1 else f" (retry {attempt - 1} of {MAX_RETRIES})"
        try:
            status_cb(f"📤 Uploading {kb} KB to {short_name}…{suffix}")
            content = _stream_completion(payload, key, status_cb, started, short_name)
            took = time.time() - started
            if not content.strip():
                # Mobile networks drop long-lived connections; that is worth
                # retrying, and worth naming accurately rather than blaming
                # the model for bad JSON.
                last_error = "The connection closed before the model sent anything."
                continue
            parsed = _extract_json(content)
            if parsed is None:
                last_error = f"{short_name} replied with something that was not JSON."
                continue
            books = parsed.get("books", [])
            status_cb(f"✅ {short_name} found {len(books)} books in {took:.1f}s")
            return books
        except urllib.error.HTTPError as e:
            last_error = _http_error_message(e)
            # 4xx other than rate-limiting will not get better on a retry.
            if e.code not in (408, 409, 429) and e.code < 500:
                break
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            last_error = f"Network/timeout: {reason}"
        except RuntimeError as e:
            last_error = str(e)
            break
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt <= MAX_RETRIES:
            backoff = 2 ** attempt
            status_cb(f"⚠️ {last_error} — retrying in {backoff}s…")
            time.sleep(backoff)

    st.error(f"API error ({short_name}): {last_error}")
    return []


def ping_api(model_id, key):
    """Cheap round-trip so you can tell a dead key from a slow model."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 15,
    }
    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode("utf-8"), headers=_api_headers(key)
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        return True, f"✅ {model_id.split('/')[-1]} reachable in {time.time() - started:.1f}s"
    except urllib.error.HTTPError as e:
        return False, _http_error_message(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {getattr(e, 'reason', e)}"


# Lazy-load OCR engine only when requested
@st.cache_resource
def get_ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(text_score=0.22)

def get_secret(name: str):
    """Recursively search st.secrets, os.environ, and .env for a secret key (case-insensitive)."""
    # 1. Recursive search in st.secrets (handles TOML nested sections like [gcp_service_account])
    try:
        if hasattr(st, "secrets"):
            def _walk(obj):
                if isinstance(obj, str):
                    return None
                items = []
                if hasattr(obj, "items"):
                    items = obj.items()
                for k, v in items:
                    if str(k).strip().lower() == name.strip().lower() and isinstance(v, str) and v.strip():
                        return v.strip().strip("\"'")
                    if hasattr(v, "items") or isinstance(v, dict):
                        found = _walk(v)
                        if found:
                            return found
                return None
            res = _walk(st.secrets)
            if res:
                return res
    except Exception:
        pass

    # 2. Check os.environ
    for env_k, env_v in os.environ.items():
        if env_k.strip().lower() == name.strip().lower() and env_v.strip():
            return env_v.strip().strip("\"'")

    # 3. Check .env file if present
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        if k.strip().lower() == name.strip().lower():
                            return v.strip().strip("\"'")
        except Exception:
            pass

    return None

def get_openrouter_key():
    return get_secret("OPENROUTER_API_KEY")

def get_gemini_key():
    for candidate in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY", "AISTUDIO_API_KEY"]:
        val = get_secret(candidate)
        if val:
            return val
    return None

def get_all_secret_keys():
    """Return list of all key names in st.secrets (names only, no values) for diagnostics."""
    keys = []
    try:
        if hasattr(st, "secrets"):
            def _collect(obj, prefix=""):
                if hasattr(obj, "items"):
                    for k, v in obj.items():
                        full_k = f"{prefix}.{k}" if prefix else str(k)
                        keys.append(full_k)
                        if hasattr(v, "items") or isinstance(v, dict):
                            _collect(v, full_k)
            _collect(st.secrets)
    except Exception:
        pass
    return keys

detected_key = get_openrouter_key()

# Sidebar: API & Model Selection
st.sidebar.header("⚡ Model & Speed Controls")

if detected_key:
    st.sidebar.success("✅ OpenRouter Key Connected")
    api_key = detected_key
else:
    api_key = st.sidebar.text_input("Enter OpenRouter API Key", type="password")

selected_model = st.sidebar.selectbox(
    "🚀 AI Vision Model",
    [
        "google/gemini-3.8-flash",        # Latest Generation + Reasoning (Anti-Hallucination) (~$0.003/scan)
    ],
    index=0,
    help="Google Gemini 3.8 Flash (Multi-step reasoning and lowest hallucination rate)."
)

scanner_mode = st.sidebar.selectbox(
    "Processing Mode",
    [
        "⚡ Fast Cloud Vision (Recommended - 2s)",
        "🔬 Hybrid (Cloud Vision + Local OCR Pass - 15s)",
        "💻 Offline OCR Only (No API Key)"
    ]
)

deduplicate_catalog = st.sidebar.checkbox(
    "🔄 Deduplicate Overlaps & Repeat Sightings", 
    value=True
)

auto_enrich_authors = st.sidebar.checkbox(
    "🌟 Auto-Enrich Catalog Intelligence Hub",
    value=True,
    help="Compares shelf against Google Sheets and offers 1-click zero-cost enrichment (Author Sales or Full Deep Dive) via Google AI Studio Gemini 3.5 Flash-Lite."
)


if st.sidebar.button("🗑️ Reset / Clear All"):
    st.session_state.processed_images = {}
    st.session_state.master_books = []
    st.session_state.pending_uploads = {}
    st.session_state.custom_shelf_dividers = {}
    st.session_state.uploader_nonce += 1
    st.session_state.last_client_batch_id = ""
    st.session_state.use_fallback_uploader = False
    st.rerun()

if st.sidebar.button("🔌 Test API Connection"):
    if not api_key:
        st.sidebar.error("No API key set.")
    else:
        ok, msg = ping_api(selected_model, api_key)
        (st.sidebar.success if ok else st.sidebar.error)(msg)

st.sidebar.markdown("---")
if is_gsheets_configured():
    sh, g_err = get_gsheet_connection()
    if sh:
        st.sidebar.success(f"🟢 **Google Sheets Connected**  \n`{sh.title}`")
        if st.sidebar.button("🔄 Sync with Google Sheets"):
            with st.sidebar.status("🔄 Syncing with Google Sheets…"):
                g_books, g_b_arch, g_a_arch, g_g_arch = load_all_from_gsheets(sh)
                if g_books:
                    for b in g_books:
                        b["author_fame_score"] = compute_author_fame_score(b.get("author_fame"), b.get("author_fame_score"))
                    st.session_state.master_books = g_books
                    save_local_master_catalog(g_books)
                if g_b_arch:
                    loc_b = load_book_archive()
                    loc_b.update(g_b_arch)
                    save_book_archive(loc_b)
                if g_a_arch:
                    loc_a = load_author_archive()
                    for k, val in g_a_arch.items():
                        val["author_fame_score"] = compute_author_fame_score(val.get("author_fame"), val.get("author_fame_score"))
                        loc_a[k] = val
                    save_author_archive(loc_a)
                if g_g_arch:
                    loc_g = load_genre_archive()
                    loc_g.update(g_g_arch)
                    save_genre_archive(loc_g)
                ok, msg = sync_catalog_to_gsheets(sh, st.session_state.master_books, get_canonical_key)
                sync_authors_to_gsheets(sh, load_author_archive())
                sync_books_to_gsheets(sh, load_book_archive())
            if ok:
                st.sidebar.success(f"✅ {msg}")
            else:
                st.sidebar.warning(f"⚠️ {msg}")
            st.rerun()
    else:
        st.sidebar.warning(f"⚠️ Google Sheets error: {g_err}")
elif not is_gsheets_available():
    st.sidebar.warning("⚠️ `gspread` not installed in environment. If on Streamlit Cloud, please **Reboot** the app to install packages.")
else:
    st.sidebar.info("⚪ Google Sheets: Not configured (saving to local disk)")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filters")


raw_cats = sorted(list(set(
    str(b.get("category") or "").strip()
    for b in st.session_state.get("master_books", [])
    if b.get("category") and b.get("category") != "-"
)))
cat_options = ["All Categories"] + raw_cats if raw_cats else [
    "All Categories", "Psychological Thriller", "Domestic Suspense", "Police Procedural",
    "Cozy Mystery", "Espionage / Action Thriller", "Sci-Fi / Fantasy", "Contemporary Romance",
    "Historical Fiction", "Literary Fiction", "General Fiction"
]
cat_filter = st.sidebar.selectbox("📖 Story Category", cat_options)


sort_by = st.sidebar.selectbox(
    "📊 Sort Master Catalog By", 
    [
        "🌟 Author Fame & Lifetime Sales (Within Genre)",
        "Sightings Count (Most Frequent First)", 
        "Author Name", 
        "Book Title"
    ],
    index=0
)

# Process Image
def downscale_ingest_bytes(img_bytes, max_dim=MAX_UPLOAD_DIM):
    """Downscale to MAX_UPLOAD_DIM at ingest.

    Prevents holding multi-megabyte raw originals in session_state,
    protects against container memory spikes, and speeds up UI reruns.
    Respects EXIF orientation so portrait phone photos stay upright.
    """
    if not img_bytes:
        return img_bytes
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        w, h = pil_img.size
        cur_max = max(w, h)
        if cur_max > max_dim:
            scale = float(max_dim) / cur_max
            new_size = (int(w * scale), int(h * scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95, optimize=True)
        return buf.getvalue()
    except Exception:
        return img_bytes


def decode_photo(img_bytes):
    """Bytes -> BGR array. Returns (image, error_message).

    Pillow goes first on purpose: it honours the EXIF orientation tag that phone
    cameras set, while cv2.imdecode ignores it and silently hands back a shelf
    rotated 90 degrees, which the model then reads badly. Pillow also covers
    HEIC once pillow_heif is installed.
    """
    if not img_bytes:
        return None, "The file arrived empty -- the upload was cut off before it finished."
    pil_error = None
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), None
    except Exception as e:
        pil_error = e

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img, None

    hint = ""
    if not HEIF_SUPPORTED:
        hint = (" If this is a HEIC photo (Pixel camera set to 'High efficiency'),"
                " install pillow-heif or switch the camera to JPEG.")
    return None, f"Could not decode this image ({pil_error}).{hint}"


def render_zoomable_image(pil_image, height=650):
    """Render an interactive pan & pinch-to-zoom image viewer in Streamlit."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=88)
    b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        background: #0e1117;
        overflow: hidden;
        width: 100%;
        height: {height}px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        position: relative;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2);
      }}
      .controls {{
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 1000;
        display: flex;
        gap: 6px;
        background: rgba(15, 17, 23, 0.85);
        backdrop-filter: blur(4px);
        padding: 6px 10px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      }}
      .btn {{
        background: #262730;
        color: #f0f2f6;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 5px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        user-select: none;
        touch-action: manipulation;
        transition: background 0.15s;
      }}
      .btn:hover {{ background: #31333F; }}
      .btn:active {{ background: #ff4b4b; border-color: #ff4b4b; }}
      .viewport {{
        position: relative;
        width: 100%;
        height: 100%;
        overflow: hidden;
        cursor: grab;
        touch-action: none;
        background: radial-gradient(circle, #1a1c24 0%, #0e1117 100%);
      }}
      .viewport:active {{ cursor: grabbing; }}
      #pan-img {{
        position: absolute;
        top: 0;
        left: 0;
        transform-origin: 0 0;
        user-select: none;
        -webkit-user-drag: none;
        pointer-events: none;
        max-width: none;
        max-height: none;
        display: block;
      }}
      .hint {{
        position: absolute;
        bottom: 8px;
        right: 12px;
        font-size: 11px;
        color: rgba(255,255,255,0.5);
        pointer-events: none;
        z-index: 100;
      }}
    </style>
    </head>
    <body>
      <div class="controls">
        <button class="btn" id="btn-in">➕ In</button>
        <button class="btn" id="btn-out">➖ Out</button>
        <button class="btn" id="btn-100">🔍 100%</button>
        <button class="btn" id="btn-reset">🔄 Fit</button>
      </div>
      <div class="hint">Pinch or scroll to zoom &bull; Drag to pan</div>
      <div class="viewport" id="vp">
        <img id="pan-img" src="data:image/jpeg;base64,{b64_data}">
      </div>

      <script>
        const vp = document.getElementById("vp");
        const img = document.getElementById("pan-img");

        let scale = 1;
        let panX = 0;
        let panY = 0;
        let isDragging = false;
        let startX = 0;
        let startY = 0;
        let pinchStartDist = 0;
        let pinchStartScale = 1;
        let pinchStartMidX = 0;
        let pinchStartMidY = 0;
        let pinchStartPanX = 0;
        let pinchStartPanY = 0;

        function updateTransform() {{
          img.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
        }}

        function zoomAround(cx, cy, factor) {{
          const newScale = Math.max(0.02, Math.min(25, scale * factor));
          const actualFactor = newScale / scale;
          panX = cx - (cx - panX) * actualFactor;
          panY = cy - (cy - panY) * actualFactor;
          scale = newScale;
          updateTransform();
        }}

        function fitImage() {{
          const vpW = vp.clientWidth;
          const vpH = vp.clientHeight;
          if (!vpW || !vpH) return;
          const imgW = img.naturalWidth || vpW;
          const imgH = img.naturalHeight || vpH;
          if (!imgW || !imgH) return;
          const pad = 10;
          const fitScale = Math.min((vpW - pad * 2) / imgW, (vpH - pad * 2) / imgH);
          scale = fitScale;
          panX = (vpW - imgW * scale) / 2;
          panY = (vpH - imgH * scale) / 2;
          updateTransform();
        }}

        let hasFitted = false;
        function tryInitialFit() {{
          if (img.naturalWidth > 0 && vp.clientWidth > 0 && vp.clientHeight > 0) {{
            fitImage();
            hasFitted = true;
          }}
        }}

        if (img.complete) {{
          tryInitialFit();
        }} else {{
          img.onload = tryInitialFit;
        }}

        const ro = new ResizeObserver(() => {{
          if (!hasFitted && vp.clientWidth > 0) {{
            tryInitialFit();
          }}
        }});
        ro.observe(vp);
        window.addEventListener("resize", fitImage);

        document.getElementById("btn-in").onclick = () => {{
          zoomAround(vp.clientWidth / 2, vp.clientHeight / 2, 1.35);
        }};
        document.getElementById("btn-out").onclick = () => {{
          zoomAround(vp.clientWidth / 2, vp.clientHeight / 2, 1 / 1.35);
        }};
        document.getElementById("btn-100").onclick = () => {{
          const vpW = vp.clientWidth;
          const vpH = vp.clientHeight;
          const imgW = img.naturalWidth || vpW;
          const imgH = img.naturalHeight || vpH;
          scale = 1.0;
          panX = (vpW - imgW) / 2;
          panY = (vpH - imgH) / 2;
          updateTransform();
        }};
        document.getElementById("btn-reset").onclick = fitImage;

        vp.addEventListener("pointerdown", (e) => {{
          if (e.pointerType === "touch" && !e.isPrimary) return;
          isDragging = true;
          startX = e.clientX - panX;
          startY = e.clientY - panY;
          vp.setPointerCapture(e.pointerId);
        }});

        vp.addEventListener("pointermove", (e) => {{
          if (!isDragging) return;
          panX = e.clientX - startX;
          panY = e.clientY - startY;
          updateTransform();
        }});

        vp.addEventListener("pointerup", (e) => {{
          isDragging = false;
          try {{ vp.releasePointerCapture(e.pointerId); }} catch(err) {{}}
        }});
        vp.addEventListener("pointercancel", () => {{ isDragging = false; }});

        vp.addEventListener("wheel", (e) => {{
          e.preventDefault();
          const rect = vp.getBoundingClientRect();
          const mouseX = e.clientX - rect.left;
          const mouseY = e.clientY - rect.top;
          const factor = e.deltaY < 0 ? 1.2 : 0.83;
          zoomAround(mouseX, mouseY, factor);
        }}, {{ passive: false }});

        vp.addEventListener("touchstart", (e) => {{
          if (e.touches.length === 2) {{
            isDragging = false;
            const t1 = e.touches[0];
            const t2 = e.touches[1];
            pinchStartDist = Math.hypot(t1.clientX - t2.clientX, t1.clientY - t2.clientY);
            pinchStartScale = scale;
            const rect = vp.getBoundingClientRect();
            pinchStartMidX = (t1.clientX + t2.clientX) / 2 - rect.left;
            pinchStartMidY = (t1.clientY + t2.clientY) / 2 - rect.top;
            pinchStartPanX = panX;
            pinchStartPanY = panY;
          }}
        }}, {{ passive: true }});

        vp.addEventListener("touchmove", (e) => {{
          if (e.touches.length === 2 && pinchStartDist > 0) {{
            e.preventDefault();
            const t1 = e.touches[0];
            const t2 = e.touches[1];
            const currentDist = Math.hypot(t1.clientX - t2.clientX, t1.clientY - t2.clientY);
            const rect = vp.getBoundingClientRect();
            const currentMidX = (t1.clientX + t2.clientX) / 2 - rect.left;
            const currentMidY = (t1.clientY + t2.clientY) / 2 - rect.top;
            
            const pinchFactor = currentDist / pinchStartDist;
            const newScale = Math.max(0.02, Math.min(25, pinchStartScale * pinchFactor));
            const actualFactor = newScale / pinchStartScale;
            
            panX = currentMidX - (pinchStartMidX - pinchStartPanX) * actualFactor;
            panY = currentMidY - (pinchStartMidY - pinchStartPanY) * actualFactor;
            scale = newScale;
            updateTransform();
          }}
        }}, {{ passive: false }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)


def estimate_tilt(crop):
    """Estimate physical slant angle of a book spine from image edges (-30 to +30 deg)."""
    h, w = crop.shape[:2]
    if h < 40 or w < 12:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=int(h * 0.3), maxLineGap=int(h * 0.1))
    if lines is None:
        return 0.0
    angles = []
    for l in lines.reshape((-1, 4)):
        x1, y1, x2, y2 = l
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dy == 0:
            continue
        deg = math.degrees(math.atan2(dx, dy))
        if deg > 90:
            deg -= 180
        elif deg < -90:
            deg += 180
        if 2.0 <= abs(deg) <= 30.0:
            angles.append(deg)
    if not angles:
        return 0.0
    return round(float(np.median(angles)), 1)


def stitch_split_books(books):
    """Re-stitch square / face-out book covers that were split across horizontal slice cuts."""
    if not books:
        return []
    merged = []
    used = set()
    for i in range(len(books)):
        if i in used:
            continue
        b1 = dict(books[i])
        box1 = list(b1.get("box_2d", [0, 0, 0, 0]))
        for j in range(i + 1, len(books)):
            if j in used:
                continue
            b2 = books[j]
            box2 = b2.get("box_2d", [0, 0, 0, 0])
            w1 = box1[3] - box1[1]
            w2 = box2[3] - box2[1]
            x_inter = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
            # Only stitch fragments from vertically adjacent shelves
            s1 = b1.get("shelf_row", 1)
            s2 = b2.get("shelf_row", 1)
            if abs(s1 - s2) == 1 and x_inter / float(max(w1, w2) + 1e-5) > 0.70:
                h1 = box1[2] - box1[0]
                h2 = box2[2] - box2[0]
                # Guard: only stitch if at least one piece is a small cut sliver fragment (< 70 units tall).
                # Two intact books on adjacent shelves (both >= 70) must never be merged.
                if min(h1, h2) < 70:
                    v_gap = max(box1[0], box2[0]) - min(box1[2], box2[2])
                    combined_h = max(box1[2], box2[2]) - min(box1[0], box2[0])
                    if v_gap <= 30 and combined_h < 260:
                        box1[0] = min(box1[0], box2[0])
                        box1[1] = min(box1[1], box2[1])
                        box1[2] = max(box1[2], box2[2])
                        box1[3] = max(box1[3], box2[3])
                        used.add(j)
                        if "Unidentified" in b1.get("title", "") and "Unidentified" not in b2.get("title", ""):
                            b1["title"] = b2.get("title")
                            b1["author"] = b2.get("author")
                            b1["spine_text"] = b2.get("spine_text")
        b1["box_2d"] = box1
        merged.append(b1)
    return merged


def apply_nms(books, iou_threshold=0.45):
    """Eliminate duplicate ghost boxes across overlapping slice cuts."""
    if not books:
        return []
    books.sort(key=lambda b: (b.get("box_2d", [0,0,0,0])[2] - b.get("box_2d", [0,0,0,0])[0]) * 
                            (b.get("box_2d", [0,0,0,0])[3] - b.get("box_2d", [0,0,0,0])[1]), reverse=True)
    kept = []
    for b in books:
        b_box = b.get("box_2d", [0, 0, 0, 0])
        is_dup = False
        for k in kept:
            k_box = k.get("box_2d", [0, 0, 0, 0])
            y1 = max(b_box[0], k_box[0])
            x1 = max(b_box[1], k_box[1])
            y2 = min(b_box[2], k_box[2])
            x2 = min(b_box[3], k_box[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter > 0:
                area_b = (b_box[2] - b_box[0]) * (b_box[3] - b_box[1])
                area_k = (k_box[2] - k_box[0]) * (k_box[3] - k_box[1])
                iou = inter / float(area_b + area_k - inter)
                containment = inter / float(min(area_b, area_k) + 1e-5)
                if iou > iou_threshold or containment > 0.65:
                    is_dup = True
                    if "Unidentified" in k.get("title", "") and "Unidentified" not in b.get("title", ""):
                        k["title"] = b.get("title")
                        k["author"] = b.get("author")
                        k["spine_text"] = b.get("spine_text")
                    break

            # Check if this box is a partial sliver fragment of an adjacent full book
            w_b = b_box[3] - b_box[1]
            w_k = k_box[3] - k_box[1]
            x_inter = max(0, min(b_box[3], k_box[3]) - max(b_box[1], k_box[1]))
            if x_inter / float(min(w_b, w_k) + 1e-5) > 0.75:
                h_b = b_box[2] - b_box[0]
                h_k = k_box[2] - k_box[0]
                v_overlap = max(0, min(b_box[2], k_box[2]) - max(b_box[0], k_box[0]))
                v_gap = max(0, max(b_box[0], k_box[0]) - min(b_box[2], k_box[2]))
                # Suppress only if there is real vertical overlap (> 10) OR one is a tiny fragment (< 40)
                if (v_overlap > 10) or (v_gap < 15 and min(h_b, h_k) < 40 and (h_b < h_k * 0.45 or h_k < h_b * 0.45)):
                    is_dup = True
                    if "Unidentified" in k.get("title", "") and "Unidentified" not in b.get("title", ""):
                        k["title"] = b.get("title")
                        k["author"] = b.get("author")
                        k["spine_text"] = b.get("spine_text")
                    break
        if not is_dup:
            kept.append(b)
    return kept


def detect_shelf_planks(img_bgr, expected_shelves=None):
    """Detect horizontal wooden shelf planks using Sobel edge filter, CLAHE, and adaptive prominence peak detection."""
    H, W = img_bgr.shape[:2]
    # Standardize detection height to 1000px for scale-invariant kernel & lighting
    det_h = 1000
    scale = det_h / float(H)
    det_w = max(50, int(W * scale))
    small = cv2.resize(img_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # CLAHE equalizes local contrast so bottom shadow shelves are as distinct as top bright shelves
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    sobel_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    sobel_abs = np.abs(sobel_y)

    kernel_w = max(20, int(det_w * 0.10))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    morph = cv2.morphologyEx(sobel_abs, cv2.MORPH_OPEN, kernel)
    row_sum = np.sum(morph, axis=1)

    k_smooth = 15
    smoothed = np.convolve(row_sum, np.ones(k_smooth) / k_smooth, mode='same')

    target_peaks = (expected_shelves - 1) if (expected_shelves and expected_shelves > 1) else None
    min_dist = int(det_h / (expected_shelves + 2)) if target_peaks else int(det_h * 0.09)

    prom_factors = [0.08, 0.05, 0.03, 0.015, 0.008] if target_peaks else [0.08]
    peaks = []
    candidates = []

    for pf in prom_factors:
        min_prom = float(np.max(smoothed)) * pf
        cands = []
        for i in range(int(det_h * 0.08), int(det_h * 0.95)):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                l = i - 1
                while l > 0 and smoothed[l] <= smoothed[i]:
                    l -= 1
                l_min = np.min(smoothed[l:i])
                r = i + 1
                while r < det_h - 1 and smoothed[r] <= smoothed[i]:
                    r += 1
                r_min = np.min(smoothed[i + 1:r + 1])
                prom = smoothed[i] - max(l_min, r_min)
                if prom >= min_prom:
                    cands.append((i, prom))

        cands.sort(key=lambda x: x[1], reverse=True)
        filt = []
        for p, prom in cands:
            if all(abs(p - f) >= min_dist for f in filt):
                filt.append(p)
        if target_peaks:
            if len(filt) >= target_peaks:
                candidates = cands
                peaks = filt
                break
        else:
            candidates = cands
            peaks = filt
            break

    if not peaks and cands:
        peaks = filt

    if target_peaks and len(peaks) > target_peaks:
        prom_dict = dict(candidates)
        sorted_peaks = sorted(peaks, key=lambda p: prom_dict.get(p, 0), reverse=True)[:target_peaks]
        peaks = sorted(sorted_peaks)
    else:
        peaks = sorted(peaks)

    # Scale peaks back to original H
    return [int(p / scale) for p in peaks]


def compute_shelf_slices(H, W, planks, pad_ratio=0.006):
    """Generate slice bounding boxes from detected horizontal shelf planks with boundary padding."""
    pad = int(H * pad_ratio)
    cuts = [0] + sorted(planks) + [H]
    slices = []
    for i in range(len(cuts) - 1):
        y1 = max(0, cuts[i] - (pad if i > 0 else 0))
        y2 = min(H, cuts[i + 1] + (pad if i < len(cuts) - 2 else 0))
        slices.append((y1, y2))
    return slices


def compute_equal_slices(H, W, num_shelves):
    """Generate equal-fraction slice bounding boxes as fallback."""
    slices = []
    for i in range(num_shelves):
        y1 = max(0, int((i / float(num_shelves) - 0.04) * H)) if i > 0 else 0
        y2 = min(H, int(((i + 1) / float(num_shelves) + 0.04) * H)) if i < num_shelves - 1 else H
        slices.append((y1, y2))
    return slices


def process_bookshelf(img_bytes, image_id, image_name, mode, model_id, key, status_cb=None, use_parallel=False, num_shelves=4, auto_detect_shelves=True, custom_dividers=None):
    status_cb = status_cb or (lambda _msg: None)
    img, decode_error = decode_photo(img_bytes)
    if img is None:
        return None, [], decode_error

    H, W, _ = img.shape
    # Downscale before upload: a 12MP phone photo is ~5 MB, and shrinking the
    # long edge is the single biggest lever on both upload time and token cost.
    max_dim = max(H, W)
    if max_dim > MAX_UPLOAD_DIM:
        scale = float(MAX_UPLOAD_DIM) / max_dim
        img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    H, W, _ = img.shape

    books_out = []
    use_api = "Offline" not in mode and bool(key)

    if use_api:
        crops_data = []
        slice_bounds = []

        if custom_dividers is not None:
            clean_divs = []
            for d in custom_dividers:
                if isinstance(d, dict):
                    yl = float(d.get("y_left", d.get("y", 0.5)))
                    yr = float(d.get("y_right", d.get("y", 0.5)))
                elif isinstance(d, (list, tuple)) and len(d) >= 2:
                    yl, yr = float(d[0]), float(d[1])
                else:
                    yl = yr = float(d)
                clean_divs.append((max(0.01, min(0.99, yl)), max(0.01, min(0.99, yr))))

            clean_divs.sort(key=lambda item: (item[0] + item[1]) / 2.0)

            # Deduplicate near-identical cuts (< 1.5%)
            dedup_divs = []
            for item in clean_divs:
                if not dedup_divs or abs(((item[0] + item[1]) / 2.0) - ((dedup_divs[-1][0] + dedup_divs[-1][1]) / 2.0)) >= 0.015:
                    dedup_divs.append(item)

            if len(dedup_divs) > 0:
                bounds = [(0.0, 0.0)] + dedup_divs + [(1.0, 1.0)]
                pad = int(H * 0.006)
                for s_idx in range(len(bounds) - 1):
                    (top_yl, top_yr) = bounds[s_idx]
                    (bot_yl, bot_yr) = bounds[s_idx + 1]

                    py_tl = max(0, int(top_yl * H) - (pad if s_idx > 0 else 0))
                    py_tr = max(0, int(top_yr * H) - (pad if s_idx > 0 else 0))
                    py_bl = min(H, int(bot_yl * H) + (pad if s_idx < len(bounds) - 2 else 0))
                    py_br = min(H, int(bot_yr * H) + (pad if s_idx < len(bounds) - 2 else 0))

                    target_h = int(max(py_bl - py_tl, py_br - py_tr))
                    if target_h < 30:
                        continue

                    # If slanted or level, warp perspective to create a leveled shelf crop
                    src_pts = np.float32([[0, py_tl], [W, py_tr], [W, py_br], [0, py_bl]])
                    dst_pts = np.float32([[0, 0], [W, 0], [W, target_h], [0, target_h]])
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

                    crop_img = cv2.warpPerspective(img, M, (W, target_h))
                    crops_data.append((s_idx + 1, crop_img, M_inv, target_h))

                status_cb(f"🪵 Prepared {len(crops_data)} leveled shelf bands from your 2-anchor lines…")

        elif use_parallel:
            if auto_detect_shelves:
                planks = detect_shelf_planks(img, expected_shelves=num_shelves if (num_shelves and num_shelves > 1) else None)
                if planks:
                    slice_bounds = compute_shelf_slices(H, W, planks)
                    status_cb(f"🪵 Auto-detected {len(slice_bounds)} physical shelves from horizontal planks! Launching parallel workers…")
            
            if not slice_bounds:
                n_s = num_shelves if (num_shelves and num_shelves > 1) else 4
                slice_bounds = compute_equal_slices(H, W, n_s)
                status_cb(f"🚀 Launching {len(slice_bounds)} parallel shelf workers simultaneously…")
                
            for s_idx, (y1, y2) in enumerate(slice_bounds, start=1):
                crops_data.append((s_idx, img[y1:y2, :], y1, y2 - y1))

        if crops_data and len(crops_data) > 1:
            started_p = time.time()

            def _worker(args):
                s_idx, crop_img, transform_info, ch = args
                res = call_vision_api(crop_img, model_id, key, status_cb=None)
                remapped = []
                for ab in res:
                    ymin, xmin, ymax, xmax = ab.get("box_2d", [0, 0, 0, 0])
                    box_h_pct = (ymax - ymin) / 1000.0
                    if (ymin <= 40 and box_h_pct < 0.35) or (ymax >= 960 and box_h_pct < 0.35):
                        continue

                    if isinstance(transform_info, np.ndarray):
                        M_inv = transform_info
                        c_ymin = (ymin / 1000.0) * ch
                        c_xmin = (xmin / 1000.0) * W
                        c_ymax = (ymax / 1000.0) * ch
                        c_xmax = (xmax / 1000.0) * W
                        c_corners = np.float32([[[c_xmin, c_ymin]], [[c_xmax, c_ymin]], [[c_xmax, c_ymax]], [[c_xmin, c_ymax]]])
                        orig_corners = cv2.perspectiveTransform(c_corners, M_inv).reshape(-1, 2)
                        orig_ymin = np.min(orig_corners[:, 1])
                        orig_ymax = np.max(orig_corners[:, 1])
                        orig_xmin = np.min(orig_corners[:, 0])
                        orig_xmax = np.max(orig_corners[:, 0])

                        ab["box_2d"] = [
                            max(0, min(1000, int((orig_ymin / float(H)) * 1000.0))),
                            max(0, min(1000, int((orig_xmin / float(W)) * 1000.0))),
                            max(0, min(1000, int((orig_ymax / float(H)) * 1000.0))),
                            max(0, min(1000, int((orig_xmax / float(W)) * 1000.0))),
                        ]
                    else:
                        y_off = transform_info
                        abs_ymin = int((ymin / 1000.0) * ch) + y_off
                        abs_ymax = int((ymax / 1000.0) * ch) + y_off
                        ab["box_2d"] = [
                            max(0, min(1000, int((abs_ymin / float(H)) * 1000.0))),
                            xmin,
                            max(0, min(1000, int((abs_ymax / float(H)) * 1000.0))),
                            xmax
                        ]

                    ab["shelf_row"] = s_idx
                    remapped.append(ab)
                return remapped

            api_books = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(crops_data)) as executor:
                future_to_shelf = {executor.submit(_worker, c): c[0] for c in crops_data}
                for future in concurrent.futures.as_completed(future_to_shelf):
                    s_num = future_to_shelf[future]
                    try:
                        shelf_books = future.result()
                        api_books.extend(shelf_books)
                    except Exception as ex:
                        st.warning(f"⚠️ Parallel worker for Shelf {s_num} failed: {ex}")

            # Re-stitch covers cut in half by slice cuts & eliminate duplicate overlaps
            api_books = stitch_split_books(api_books)
            api_books = apply_nms(api_books, iou_threshold=0.45)
            status_cb(f"✅ Parallel scan finished in {time.time() - started_p:.1f}s — {len(api_books)} unique books found across {len(crops_data)} shelves!")
        else:
            api_books = call_vision_api(img, model_id, key, status_cb)
        
        # Ensure strict top-to-bottom, left-to-right ordering across shelves
        def _get_sort_key(ab):
            ymin, xmin, _, _ = ab.get("box_2d", [0, 0, 0, 0])
            return (ab.get("shelf_row", 1), xmin)
        
        api_books.sort(key=_get_sort_key)

        for idx, ab in enumerate(api_books, start=1):
            ymin, xmin, ymax, xmax = ab.get("box_2d", [0, 0, 0, 0])
            px_ymin = int((ymin / 1000.0) * H)
            px_xmin = int((xmin / 1000.0) * W)
            px_ymax = int((ymax / 1000.0) * H)
            px_xmax = int((xmax / 1000.0) * W)
            
            tilt = float(ab.get("tilt_angle", 0) or 0)
            # Physical edge analysis: if book is tilted on shelf, detect angle directly from pixels
            if abs(tilt) < 1.0 and (px_ymax - px_ymin) > 40 and (px_xmax - px_xmin) > 12:
                crop = img[px_ymin:px_ymax, px_xmin:px_xmax]
                tilt = estimate_tilt(crop)
            tilt = max(-35.0, min(35.0, tilt))
            cx = (px_xmin + px_xmax) / 2.0
            cy = (px_ymin + px_ymax) / 2.0
            w = max(4.0, float(px_xmax - px_xmin))
            h = max(4.0, float(px_ymax - px_ymin))
            rect = ((cx, cy), (w, h), tilt)
            pts = np.int32(cv2.boxPoints(rect)).reshape((-1, 1, 2))
            
            books_out.append({
                "id": idx,
                "image_id": image_id,
                "image_name": image_name,
                "shelf": ab.get("shelf_row", 1),
                "box_2d": [int(ymin), int(xmin), int(ymax), int(xmax)],
                "title": str(ab.get("title") or f"Book {idx}"),
                "author": str(ab.get("author") or "Unknown"),
                "spine_text": str(ab.get("spine_text") or ""),
                "category": ab.get("category", "Standalone Novel"),
                "series": ab.get("series_info", "-"),
                "protagonist": ab.get("protagonist", "-"),
                "sensual_romance_flag": "-",
                "tv_adaptation": "-",
                "sales": "-",
                "deep_searched": False,
                "search_evidence": "-",
                "box_pixels": [px_xmin, px_ymin, px_xmax, px_ymax],
                "polygon_pts": pts,
                "source": model_id.split("/")[-1]
            })
            
    # Pure offline fallback or explicit hybrid.
    # "Hybrid" used to fall through here without matching, so the local pass it
    # promises never actually ran.
    if "Offline" in mode or "Hybrid" in mode or not use_api:
        if not use_api and "Offline" not in mode:
            status_cb("ℹ️ No API key, so falling back to the local OCR pass.")
        status_cb("🔎 Running the local OCR pass (this is the slow one)…")
        try:
            ocr = get_ocr_engine()
            raw_results, _ = ocr(img)
        except Exception as e:
            reason = f"Local OCR is unavailable ({type(e).__name__}: {e})."
            if not books_out:
                # It was the only source of results, so there is nothing to show.
                return None, [], reason + " Add an OpenRouter API key to use cloud vision."
            st.warning(f"⚠️ {reason} Showing cloud vision results only.")
            raw_results = None
        if raw_results:
            for idx, item in enumerate(raw_results[:60], start=len(books_out)+1):
                poly = np.array(item[0], dtype=np.int32)
                rect = cv2.minAreaRect(poly)
                pts = np.int32(cv2.boxPoints(rect)).reshape((-1, 1, 2))
                xmin, ymin = int(np.min(poly[:,0])), int(np.min(poly[:,1]))
                xmax, ymax = int(np.max(poly[:,0])), int(np.max(poly[:,1]))
                books_out.append({
                    "id": idx,
                    "image_id": image_id,
                    "image_name": image_name,
                    "shelf": 1,
                    "title": item[1][:30],
                    "author": "Offline OCR",
                    "category": "Standalone Novel",
                    "series": "-",
                    "protagonist": "-",
                    "sensual_romance_flag": "-",
                    "tv_adaptation": "-",
                    "sales": "-",
                    "deep_searched": False,
                    "search_evidence": "-",
                    "box_pixels": [xmin, ymin, xmax, ymax],
                    "polygon_pts": pts,
                    "source": "Offline OCR"
                })

    # Render Annotated Overlay
    h_img, w_img = img.shape[:2]
    scale_factor = max(1.0, h_img / 1200.0)
    box_thickness = max(2, int(2.0 * scale_factor))
    font_scale = 0.38 * scale_factor
    font_thick = max(1, int(1.2 * scale_factor))

    annotated = img.copy()
    overlay = img.copy()
    colors = [(50, 220, 100), (240, 150, 40), (220, 60, 220), (30, 200, 240), (255, 100, 50), (100, 100, 255)]
    
    # 1. Translucent shelf tint using oriented polygons
    for b in books_out:
        pts = b["polygon_pts"]
        col = colors[(b["shelf"] - 1) % len(colors)]
        cv2.fillPoly(overlay, [pts], col)

    cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)

    # 2. Crisp borders & high-contrast dynamic ID badges
    for b in books_out:
        pts = b["polygon_pts"]
        col = colors[(b["shelf"] - 1) % len(colors)]
        cv2.polylines(annotated, [pts], isClosed=True, color=col, thickness=box_thickness)
        
        text = str(b["id"])
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        pad_x = int(6 * scale_factor)
        pad_y = int(4 * scale_factor)
        badge_w = tw + pad_x * 2
        badge_h = th + pad_y * 2
        
        pts_2d = pts.reshape((-1, 2))
        top_idx = int(np.argmin(pts_2d[:, 1]))
        top_x, top_y = int(pts_2d[top_idx, 0]), int(pts_2d[top_idx, 1])
        
        badge_x1 = max(0, min(w_img - badge_w - 1, top_x))
        badge_y1 = max(0, top_y - badge_h) if top_y >= badge_h else top_y
        badge_x2 = min(w_img - 1, badge_x1 + badge_w)
        badge_y2 = min(h_img - 1, badge_y1 + badge_h)
        
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (15, 15, 15), -1)
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), col, max(1, int(1.0 * scale_factor)))
        
        text_x = badge_x1 + pad_x
        text_y = badge_y1 + th + pad_y
        cv2.putText(annotated, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
                    
    pil_res = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    return pil_res, books_out, None

def get_canonical_key(title, author):
    clean_t = re.sub(r'[^a-zA-Z0-9]', '', str(title or "").lower())
    clean_a = re.sub(r'[^a-zA-Z0-9]', '', str(author or "").lower())
    return f"{clean_t}_{clean_a}"







def execute_model_search(model_id, title, author, api_key=None):
    """Execute search using Direct Google AI Studio Gemini 3.5 Flash-Lite ($0.00)."""
    gemini_key = get_gemini_key()
    res = query_direct_gemini_api(title, author, gemini_key, preferred_model="gemini-3.5-flash-lite")
    return {
        "book_sales": res.get("book_sales", "Not publicly reported"),
        "author_fame": res.get("author_fame", "Not publicly reported"),
        "author_fame_score": res.get("author_fame_score", 0),
        "tv_adaptation": res.get("tv_deal", "No"),
        "sensual_rating": res.get("sensual_rating", "Clean / None"),
        "evidence": res.get("evidence", "-"),
        "latency": round(res.get("latency_ms", 0) / 1000.0, 2)
    }


def classify_genres_in_batch(books, api_key=None, status_cb=None, batch_size=35):
    """Classify detected books into specific genres, series, and protagonists using direct Gemini 3.5 Flash-Lite ($0 search fee). Checks genre_archive first ($0.00 / 0ms)."""
    if not books:
        return books

    status_cb = status_cb or (lambda _msg: None)
    gemini_key = get_gemini_key()

    # 1. Map canonical key to unique title & author
    unique_items = []
    seen_keys = set()
    for b in books:
        t = (b.get("title") or "").strip()
        a = (b.get("author") or "").strip()
        if not t or "unidentified" in t.lower() or t.lower().startswith("book "):
            continue
        c_key = get_canonical_key(t, a)
        if c_key not in seen_keys:
            seen_keys.add(c_key)
            unique_items.append({"key": c_key, "title": t, "author": a})

    if not unique_items:
        return books

    # 2. Check local genre cache
    genre_cache = load_genre_archive()
    results_map = {}
    items_to_classify = []

    for item in unique_items:
        c_key = item["key"]
        cached = genre_cache.get(c_key)
        if cached and cached.get("category") and cached.get("category") != "-":
            results_map[c_key] = {
                "category": cached.get("category", "General Fiction"),
                "series": cached.get("series", "Standalone Novel"),
                "protagonist": cached.get("protagonist", "-")
            }
        else:
            items_to_classify.append(item)

    if not items_to_classify:
        status_cb(f"⚡ Loaded genres & series for all {len(unique_items)} books from cache ($0.00 / 0ms)!")
    elif not gemini_key:
        status_cb("ℹ️ No GEMINI_API_KEY available for genre classification; using defaults.")
    else:
        cached_count = len(unique_items) - len(items_to_classify)
        cached_msg = f" ({cached_count} loaded from cache)" if cached_count > 0 else ""
        status_cb(f"📖 Classifying genres & series for {len(items_to_classify)} new books{cached_msg} (Gemini 3.5 Flash-Lite • $0.00)...")

        # Chunk into batches of 35
        batches = [items_to_classify[i:i + batch_size] for i in range(0, len(items_to_classify), batch_size)]

        def _process_genre_batch(batch_slice):
            items_str = "\n".join([f"{idx+1}. '{item['title']}' by '{item['author']}'" for idx, item in enumerate(batch_slice)])
            prompt = f"""You are an expert book cataloging and literature taxonomy agent.
For each of the following published books, identify:
1. "category": Primary literary fiction or non-fiction genre (e.g. 'Psychological Thriller', 'Domestic Suspense', 'Police Procedural', 'Cozy Mystery', 'Espionage / Action Thriller', 'Sci-Fi / Fantasy', 'Contemporary Romance', 'Historical Fiction', 'Literary Fiction', 'True Crime / Non-Fiction', or 'General Fiction').
2. "series": If this book is part of an established book series, provide the canonical series name (e.g. 'Chief Inspector Gamache', 'Harry Hole', 'In Death', 'Thursday Murder Club', 'Gabriel Allon', 'Jack Reacher'). If it is a standalone novel, return 'Standalone Novel'.
3. "protagonist": The primary lead character or detective name (e.g. 'Armand Gamache', 'Harry Hole', 'Eve Dallas', 'Gabriel Allon') or '-' if ensemble/standalone.

Input Books:
{items_str}

Return strictly a valid JSON object:
{{
  "books": [
    {{
      "index": 1,
      "title": "...",
      "author": "...",
      "category": "...",
      "series": "...",
      "protagonist": "..."
    }}
  ]
}}"""
            clean_gkey = str(gemini_key).strip().strip("\"'")
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": clean_gkey
            }
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingLevel": "minimal"}
                }
            }
            try:
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                raw_body, err = _execute_with_rate_limit_retry(req, max_retries=3)
                if raw_body:
                    extracted = _extract_text_from_resp(raw_body)
                    parsed = _parse_json_result(extracted) if extracted else {}
                    raw_list = parsed.get("books", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
                    batch_res = {}
                    for idx, item in enumerate(batch_slice):
                        matched = None
                        if idx < len(raw_list) and isinstance(raw_list[idx], dict):
                            matched = raw_list[idx]
                        else:
                            for entry in raw_list:
                                if isinstance(entry, dict) and item["title"].lower() in str(entry.get("title", "")).lower():
                                    matched = entry
                                    break
                        if matched:
                            batch_res[item["key"]] = {
                                "category": matched.get("category") or "General Fiction",
                                "series": matched.get("series") or "Standalone Novel",
                                "protagonist": matched.get("protagonist") or "-"
                            }
                        else:
                            batch_res[item["key"]] = {
                                "category": "General Fiction",
                                "series": "Standalone Novel",
                                "protagonist": "-"
                            }
                    return batch_res
            except Exception:
                pass
            return {item["key"]: {"category": "General Fiction", "series": "Standalone Novel", "protagonist": "-"} for item in batch_slice}

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
            future_map = {executor.submit(_process_genre_batch, bch): bch for bch in batches}
            for fut in concurrent.futures.as_completed(future_map):
                try:
                    res = fut.result()
                    results_map.update(res)
                    genre_cache.update(res)
                except Exception:
                    pass

        # Save newly classified items to disk cache
        save_genre_archive(genre_cache)

    # 3. Merge into books list
    for b in books:
        t = (b.get("title") or "").strip()
        a = (b.get("author") or "").strip()
        c_key = get_canonical_key(t, a)
        if c_key in results_map:
            info = results_map[c_key]
            b["category"] = info["category"]
            b["series"] = info["series"]
            b["protagonist"] = info["protagonist"]
        else:
            if not b.get("category"):
                b["category"] = "General Fiction"
            if not b.get("series"):
                b["series"] = "Standalone Novel"
            if not b.get("protagonist"):
                b["protagonist"] = "-"

    if items_to_classify:
        status_cb(f"✅ Classified {len(items_to_classify)} new books in ~1s (saved to cache)")
    return books


def attach_cached_data_to_books(books):
    """Attach already cached author and book data from local archives without calling any API."""
    if not books:
        return books
    archive = load_author_archive()
    book_archive = load_book_archive()
    for b in books:
        a = (b.get("author") or "").strip()
        clean_key = a.lower()
        if clean_key in archive:
            entry = archive[clean_key]
            fame_val = entry.get("author_fame", "Not publicly reported")
            b["author_fame"] = fame_val
            b["author_fame_score"] = compute_author_fame_score(fame_val, entry.get("author_fame_score", 0.0))
        else:
            if not b.get("author_fame"):
                b["author_fame"] = "Not publicly reported"
                b["author_fame_score"] = 0.0

        t = (b.get("title") or "").strip()
        b_key = get_canonical_key(t, a)
        if b_key in book_archive:
            cached_b = book_archive[b_key]
            b["sales"] = cached_b.get("book_sales", "-")
            b["tv_adaptation"] = cached_b.get("tv_adaptation", "-")
            b["sensual_romance_flag"] = cached_b.get("sensual_rating", "-")
            b["search_evidence"] = cached_b.get("evidence", "-")
            b["deep_searched"] = bool(b["sales"] and b["sales"] != "-")
        else:
            b["deep_searched"] = False
            if "sales" not in b or not b["sales"]:
                b["sales"] = "-"
            if "tv_adaptation" not in b or not b["tv_adaptation"]:
                b["tv_adaptation"] = "-"
            if "sensual_romance_flag" not in b or not b["sensual_romance_flag"]:
                b["sensual_romance_flag"] = "-"
    return books


def search_author_fame(author, api_key=None):
    """Search lifetime career sales or reader reach for an author using direct Gemini 3.5 Flash-Lite ($0.00)."""
    gemini_key = get_gemini_key()
    res = query_direct_gemini_author_fame(author, gemini_key, preferred_model="gemini-3.5-flash-lite")
    return {
        "author_fame": res.get("author_fame", "Not publicly reported"),
        "author_fame_score": res.get("author_fame_score", 0),
        "evidence": res.get("evidence", "-"),
        "latency": round(res.get("latency_ms", 0) / 1000.0, 2)
    }


def enrich_authors_in_parallel(books, api_key=None, status_cb=None, max_workers=10):
    """Enrich detected books with Author Fame for unique authors only via direct Gemini 3.5 Flash-Lite ($0.00)."""
    if not books:
        return books

    status_cb = status_cb or (lambda _msg: None)
    gemini_key = get_gemini_key()
    archive = load_author_archive()

    # Find unique authors
    unique_authors = set()
    for b in books:
        a = (b.get("author") or "").strip()
        if a and "unknown" not in a.lower() and a.lower() != "author" and not a.lower().startswith("book "):
            unique_authors.add(a)

    if unique_authors and gemini_key:
        to_query = []
        for a in unique_authors:
            clean_key = a.lower()
            if clean_key not in archive or not archive[clean_key].get("author_fame") or archive[clean_key].get("author_fame") in ["-", "Not publicly reported", ""]:
                to_query.append(a)

        cached_count = len(unique_authors) - len(to_query)
        status_cb(f"🌟 Author Archive: {cached_count} found in cache ($0.00), {len(to_query)} new to search…")

        if to_query:
            done = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_author = {
                    executor.submit(query_direct_gemini_author_fame, a, gemini_key, "gemini-3.5-flash-lite"): a
                    for a in to_query
                }
                for future in concurrent.futures.as_completed(future_to_author):
                    a = future_to_author[future]
                    try:
                        res = future.result()
                        fame = res.get("author_fame", "Not publicly reported")
                        fame_score = compute_author_fame_score(fame, res.get("author_fame_score"))
                        archive[a.lower()] = {
                            "author": a,
                            "author_fame": fame,
                            "author_fame_score": float(fame_score),
                            "evidence": res.get("evidence", "-")
                        }
                    except Exception as ex:
                        archive[a.lower()] = {
                            "author": a,
                            "author_fame": "Not publicly reported",
                            "author_fame_score": 0.0,
                            "evidence": str(ex)
                        }
                    done += 1
                    if done % 3 == 0 or done == len(to_query):
                        status_cb(f"🌟 Enriched {done}/{len(to_query)} new authors with career sales ($0.00)…")

            save_author_archive(archive)

    # Attach cached values
    return attach_cached_data_to_books(books)


def deep_search_single_book(title, author, api_key=None):
    """Execute Option 2 deep search for a single book on demand ($0.00 via Gemini 3.5 Flash-Lite)."""
    canon_key = get_canonical_key(title, author)
    book_archive = load_book_archive()
    if canon_key in book_archive and book_archive[canon_key].get("book_sales") and book_archive[canon_key].get("book_sales") != "-":
        return book_archive[canon_key]

    gemini_key = get_gemini_key()
    res = query_direct_gemini_api(title, author, gemini_key, preferred_model="gemini-3.5-flash-lite")
    
    b_data = {
        "book_sales": res.get("book_sales", "Not publicly reported"),
        "tv_adaptation": res.get("tv_deal", "No"),
        "sensual_rating": res.get("sensual_rating", "Clean / None"),
        "evidence": res.get("evidence", "-"),
        "author_fame": res.get("author_fame", "Not publicly reported"),
        "author_fame_score": compute_author_fame_score(res.get("author_fame"), res.get("author_fame_score", 0)),
        "latency": round(res.get("latency_ms", 0) / 1000.0, 2)
    }

    # Save to book archive
    book_archive[canon_key] = b_data
    save_book_archive(book_archive)

    # If author fame was found, update author archive
    if b_data["author_fame"] != "Not publicly reported":
        author_archive = load_author_archive()
        clean_author = author.strip().lower()
        if clean_author not in author_archive or author_archive[clean_author].get("author_fame") in ["-", "Not publicly reported", ""]:
            author_archive[clean_author] = {
                "author": author.strip(),
                "author_fame": b_data["author_fame"],
                "author_fame_score": float(b_data["author_fame_score"]),
                "evidence": b_data["evidence"]
            }
            save_author_archive(author_archive)

    # Sync to Google Sheets
    if is_gsheets_configured():
        sh_sync, _ = get_gsheet_connection()
        if sh_sync:
            sync_book_search_to_gsheets(sh_sync, canon_key, title, author, b_data)
            sync_authors_to_gsheets(sh_sync, load_author_archive())

    return b_data


# Startup Auto-Load from Local Disk & Google Sheets
if "catalog_restored_once" not in st.session_state:
    st.session_state.catalog_restored_once = True
    if not st.session_state.master_books:
        # 1. Local disk auto-load
        loc_saved = load_local_master_catalog()
        if loc_saved:
            st.session_state.master_books = loc_saved

        # 2. Google Sheets cloud auto-load
        if is_gsheets_configured():
            sh_boot, _ = get_gsheet_connection()
            if sh_boot:
                g_books, g_b_arch, g_a_arch, g_g_arch = load_all_from_gsheets(sh_boot)
                if g_books:
                    st.session_state.master_books = g_books
                    save_local_master_catalog(g_books)
                if g_b_arch:
                    loc_b = load_book_archive()
                    loc_b.update(g_b_arch)
                    save_book_archive(loc_b)
                if g_a_arch:
                    loc_a = load_author_archive()
                    loc_a.update(g_a_arch)
                    save_author_archive(loc_a)
                if g_g_arch:
                    loc_g = load_genre_archive()
                    loc_g.update(g_g_arch)
                    save_genre_archive(loc_g)


tab_scanner, tab_arena = st.tabs(["📸 Bookshelf Scanner & Catalog", "⚔️ Gemini 3.8 Flash Vision Arena"])
with tab_scanner:
    # Main Upload Area
    st.markdown("### 📸 Select Bookshelf Photos")

    if st.session_state.get("use_fallback_uploader", False) or _client_uploader is None:
        st.info("ℹ️ Using standard file uploader (supports HEIC/RAW).")
        uploaded_file = st.file_uploader(
            "Upload bookshelf photo",
            type=UPLOAD_TYPES,
            accept_multiple_files=False,
            key=f"shelf_uploader_{st.session_state.uploader_nonce}",
            help="Pick a shelf photo to add to your queue below."
        )
        if uploaded_file is not None:
            k = f"{uploaded_file.name}:{uploaded_file.size}"
            if k not in st.session_state.pending_uploads:
                st.session_state.pending_uploads[k] = (uploaded_file.name, downscale_ingest_bytes(uploaded_file.getvalue()))
        if st.button("⚡ Switch back to Fast Ingest"):
            st.session_state.use_fallback_uploader = False
            st.session_state.uploader_nonce += 1
            st.rerun()
    else:
        upload_data = _client_uploader(key=f"client_up_{st.session_state.uploader_nonce}")
        if upload_data and isinstance(upload_data, dict):
            if upload_data.get("need_fallback"):
                st.session_state.use_fallback_uploader = True
                st.rerun()
            batch_id = upload_data.get("batch_id")
            if batch_id and batch_id != st.session_state.get("last_client_batch_id"):
                st.session_state.last_client_batch_id = batch_id
                for f in upload_data.get("files", []):
                    name = f.get("name", "shelf.jpg")
                    durl = f.get("data", "")
                    b64_str = durl.split(",", 1)[1] if "," in durl else durl
                    try:
                        bts = base64.b64decode(b64_str)
                        k = f"{name}:{len(bts)}"
                        if k not in st.session_state.pending_uploads:
                            st.session_state.pending_uploads[k] = (name, bts)
                    except Exception:
                        pass
                st.rerun()

    with st.expander("📸 Or snap with Live Camera"):
        use_camera = st.checkbox("Turn on camera hardware", value=False, key="activate_live_camera")
        if use_camera:
            camera_photo = st.camera_input("Take shelf photo", key="shelf_camera_input")
            if camera_photo is not None:
                cam_bytes = camera_photo.getvalue()
                if cam_bytes:
                    cam_name = f"Camera_Shelf_{len(st.session_state.pending_uploads) + 1}.jpg"
                    cam_key = f"{cam_name}:{len(cam_bytes)}"
                    if cam_key not in st.session_state.pending_uploads:
                        st.session_state.pending_uploads[cam_key] = (cam_name, downscale_ingest_bytes(cam_bytes))

    # Queued Photos Display & Actions
    queued_keys = list(st.session_state.pending_uploads.keys())
    queued = [st.session_state.pending_uploads[k] for k in queued_keys]

    if queued:
        st.markdown(f"#### 📁 Queued Shelf Photos ({len(queued)} ready)")
        for k in queued_keys:
            name, bts = st.session_state.pending_uploads[k]
            qcol1, qcol2, qcol3 = st.columns([1, 4, 1])
            with qcol1:
                try:
                    st.image(bts, width=70)
                except Exception:
                    st.write("📷")
            with qcol2:
                st.write(f"**{name}** ({len(bts) // 1024} KB)")
            with qcol3:
                if st.button("✕ Remove", key=f"del_{k}"):
                    st.session_state.pending_uploads.pop(k, None)
                    st.session_state.custom_shelf_dividers.pop(k, None)
                    st.session_state.uploader_nonce += 1
                    st.session_state.last_client_batch_id = ""
                    st.rerun()

        # Shelf Boundary Inspector & Pinpointer (Option 2)
        if _shelf_pinpointer is not None:
            with st.expander("🪵 **Interactive Shelf Pinpointer (Tap photo to Add / Move Shelves)**", expanded=True):
                st.caption("OpenCV auto-detected initial shelves. **Tap directly on photo** to add any missed shelf, **drag ↕** to align on wood plank, or **tap ✕** to remove.")
                if len(queued) > 1:
                    pin_choice_idx = st.selectbox(
                        "Select Photo to Pinpoint Shelves",
                        range(len(queued)),
                        format_func=lambda i: f"Photo {i+1}: {queued[i][0]}"
                    )
                else:
                    pin_choice_idx = 0

                pin_name, pin_bts = queued[pin_choice_idx]
                pin_key = queued_keys[pin_choice_idx]

                # Compute OpenCV auto-detected planks and multi-shelf presets
                pin_img, _ = decode_photo(pin_bts)
                pin_presets = {}
                if pin_img is not None:
                    H_pin = pin_img.shape[0]
                    for n_shelf in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
                        p_peaks = detect_shelf_planks(pin_img, expected_shelves=n_shelf)
                        pin_presets[n_shelf] = [round(float(p) / H_pin, 3) for p in sorted(p_peaks)]

                if pin_key not in st.session_state.custom_shelf_dividers:
                    if pin_img is not None:
                        H_pin = pin_img.shape[0]
                        planks = detect_shelf_planks(pin_img, expected_shelves=None)
                        if planks:
                            initial_divs = [{"y_left": round(float(p) / H_pin, 3), "y_right": round(float(p) / H_pin, 3)} for p in sorted(planks)]
                        elif pin_presets.get(4):
                            initial_divs = [{"y_left": p, "y_right": p} for p in pin_presets[4]]
                        else:
                            initial_divs = [
                                {"y_left": 0.25, "y_right": 0.25},
                                {"y_left": 0.50, "y_right": 0.50},
                                {"y_left": 0.75, "y_right": 0.75}
                            ]
                    else:
                        initial_divs = [
                            {"y_left": 0.25, "y_right": 0.25},
                            {"y_left": 0.50, "y_right": 0.50},
                            {"y_left": 0.75, "y_right": 0.75}
                        ]
                    st.session_state.custom_shelf_dividers[pin_key] = initial_divs

                cur_divs = st.session_state.custom_shelf_dividers.get(pin_key, [0.25, 0.50, 0.75])

                pin_b64 = base64.b64encode(pin_bts).decode("utf-8")
                pin_event = _shelf_pinpointer(
                    image_b64=pin_b64,
                    initial_dividers=cur_divs,
                    plank_presets=pin_presets,
                    key=f"pinpointer_{pin_key}"
                )
                if pin_event and isinstance(pin_event, dict):
                    updated_divs = pin_event.get("dividers")
                    if updated_divs is not None and updated_divs != cur_divs:
                        st.session_state.custom_shelf_dividers[pin_key] = updated_divs

        action_col1, action_col2 = st.columns([3, 1])
        with action_col1:
            run_scan = st.button(
                f"🚀 Run Scanner & Identify Books ({len(queued)} Photo{'s' if len(queued) > 1 else ''})", 
                type="primary", 
                width="stretch"
            )
        with action_col2:
            if st.button("🗑️ Clear Queue", width="stretch"):
                st.session_state.pending_uploads = {}
                st.session_state.custom_shelf_dividers = {}
                st.session_state.uploader_nonce += 1
                st.session_state.last_client_batch_id = ""
                st.rerun()

        if run_scan:
            st.session_state.processed_images = {}
            st.session_state.master_books = []

            progress = st.progress(0.0)
            status = st.empty()
            run_started = time.time()
            failures = []

            for idx, (name, img_bytes) in enumerate(queued, start=1):
                def status_cb(msg, _i=idx, _name=name):
                    # Writing to the placeholder on every update is what makes the
                    # wait legible -- and the traffic doubles as a websocket
                    # keepalive, so a long call does not look like a frozen page.
                    status.info(f"**Photo {_i} of {len(queued)} — {_name}**\n\n{msg}")

                status_cb("📥 Decoding photo…")
                k_cur = queued_keys[idx - 1]
                custom_divs = st.session_state.custom_shelf_dividers.get(k_cur)
                pil_img, books, err = process_bookshelf(
                    img_bytes, idx, f"Image {idx}", scanner_mode,
                    selected_model, api_key, status_cb,
                    custom_dividers=custom_divs,
                )
                if err:
                    failures.append(f"**{name}** — {err}")
                elif pil_img is not None:
                    # Anchor landmark book extraction for Google Pixel / GSheets location
                    anchor_title = None
                    for b in books:
                        t = (b.get("title") or "").strip()
                        if t and not t.lower().startswith("book ") and "unidentified" not in t.lower():
                            a = (b.get("author") or "").strip()
                            anchor_title = f"{t} ({a})" if a and a != "Unknown" else t
                            break
                    anchor_tag = f" [Anchor: {anchor_title}]" if anchor_title else ""
                    photo_label = f"{name}{anchor_tag}"
                    for b in books:
                        b["image_name"] = photo_label

                    st.session_state.processed_images[f"Image {idx} ({name})"] = pil_img
                    st.session_state.master_books.extend(books)
                progress.progress(idx / len(queued))

            if st.session_state.master_books:
                status.info("📖 Classifying book genres & series with Gemini 3.5 Flash-Lite ($0 search fee)…")
                st.session_state.master_books = classify_genres_in_batch(
                    st.session_state.master_books,
                    status_cb=lambda msg: status.info(f"📖 **Genre Classifier**\n\n{msg}")
                )
                st.session_state.master_books = attach_cached_data_to_books(st.session_state.master_books)

            status.empty()
            progress.empty()
            for failure in failures:
                st.error(f"❌ {failure}")

            # Auto-save local disk JSON and Google Sheets
            save_local_master_catalog(st.session_state.master_books)
            if is_gsheets_configured():
                sh_sync, _ = get_gsheet_connection()
                if sh_sync:
                    ok_s, s_msg = sync_catalog_to_gsheets(sh_sync, st.session_state.master_books, get_canonical_key)
                    sync_authors_to_gsheets(sh_sync, load_author_archive())
                    sync_books_to_gsheets(sh_sync, load_book_archive())
                    if not ok_s:
                        st.warning(f"⚠️ Google Sheets sync warning: {s_msg}")

            st.success(
                f"🎉 Finished in {time.time() - run_started:.1f}s — "
                f"{len(st.session_state.master_books)} books across "
                f"{len(queued) - len(failures)} of {len(queued)} photo(s)."
            )

    # Pre-bundled demo button
    st.markdown("---")
    demo_c1, demo_c2 = st.columns([3, 2])
    with demo_c1:
        run_demo = st.button("🧪 Test with Example Shelf (Pre-bundled)", width="stretch")
    with demo_c2:
        pin_demo = st.button("🪵 Adjust Shelves on Example Shelf", width="stretch")

    if pin_demo:
        if os.path.exists(SAMPLE_IMAGE):
            with open(SAMPLE_IMAGE, "rb") as f:
                d_bytes = f.read()
            st.session_state.pending_uploads["demo_sample_shelf"] = ("sample_shelf.jpg", d_bytes)
            st.rerun()

    if run_demo:
        if os.path.exists(SAMPLE_IMAGE):
            with open(SAMPLE_IMAGE, "rb") as f:
                demo_bytes = f.read()
            status = st.empty()
            custom_demo_divs = st.session_state.custom_shelf_dividers.get("demo_sample_shelf")
            pil_img, books, err = process_bookshelf(
                demo_bytes, 1, "Image 1 (Example Bookstore Shelf)", scanner_mode,
                selected_model, api_key,
                lambda msg: status.info(f"**Example shelf**\n\n{msg}"),
                custom_dividers=custom_demo_divs,
            )
            if err:
                status.empty()
                st.error(f"❌ Example shelf — {err}")
            else:
                anchor_title = None
                for b in books:
                    t = (b.get("title") or "").strip()
                    if t and not t.lower().startswith("book ") and "unidentified" not in t.lower():
                        a = (b.get("author") or "").strip()
                        anchor_title = f"{t} ({a})" if a and a != "Unknown" else t
                        break
                anchor_tag = f" [Anchor: {anchor_title}]" if anchor_title else ""
                photo_label = f"sample_shelf.jpg{anchor_tag}"
                for b in books:
                    b["image_name"] = photo_label

                st.session_state.processed_images[f"Image 1 ({photo_label})"] = pil_img
                st.session_state.master_books = books
                if st.session_state.master_books:
                    status.info("📖 Classifying book genres & series with Gemini 3.5 Flash-Lite ($0 search fee)…")
                    st.session_state.master_books = classify_genres_in_batch(
                        st.session_state.master_books,
                        status_cb=lambda msg: status.info(f"📖 **Genre Classifier**\n\n{msg}")
                    )
                    st.session_state.master_books = attach_cached_data_to_books(st.session_state.master_books)
                status.empty()
                save_local_master_catalog(st.session_state.master_books)
                if is_gsheets_configured():
                    sh_sync, _ = get_gsheet_connection()
                    if sh_sync:
                        ok_d, d_msg = sync_catalog_to_gsheets(sh_sync, st.session_state.master_books, get_canonical_key)
                        sync_authors_to_gsheets(sh_sync, load_author_archive())
                        sync_books_to_gsheets(sh_sync, load_book_archive())
                        if not ok_d:
                            st.warning(f"⚠️ Google Sheets sync warning: {d_msg}")
                st.success(f"🎉 Example shelf loaded: {len(st.session_state.master_books)} books identified & classified!")
        else:
            st.error(f"❌ Example image is missing from the repo: {SAMPLE_IMAGE}")

    # Deduplication & Master Catalog
    raw_books = st.session_state.master_books

    if deduplicate_catalog and raw_books:
        unique_dict = {}
        for b in raw_books:
            key = get_canonical_key(b.get("title", ""), b.get("author", ""))
            loc_str = f"{b.get('image_name')} (Shelf {b.get('shelf')}, Book #{b.get('id')})"
        
            if key not in unique_dict:
                entry = dict(b)
                entry["sightings_count"] = 1
                entry["all_locations"] = [loc_str]
                unique_dict[key] = entry
            else:
                unique_dict[key]["sightings_count"] += 1
                if loc_str not in unique_dict[key]["all_locations"]:
                    unique_dict[key]["all_locations"].append(loc_str)
        display_catalog = list(unique_dict.values())
    else:
        display_catalog = []
        for b in raw_books:
            entry = dict(b)
            entry["sightings_count"] = 1
            entry["all_locations"] = [f"{b.get('image_name')} (Shelf {b.get('shelf')}, Book #{b.get('id')})"]
            display_catalog.append(entry)

    if not display_catalog:
        st.info(
            "👆 Tap **Upload** to pick pictures from your phone gallery, "
            "then press **🚀 Run Scanner**. Keep this tab open while a photo "
            "uploads — leaving the browser can cut the transfer short."
        )
    else:
        # Filter
        filtered_books = []
        for b in display_catalog:
            if cat_filter != "All Categories" and b.get("category", "") != cat_filter:
                continue
            filtered_books.append(b)

        # Sort
        if sort_by == "🌟 Author Fame & Lifetime Sales (Within Genre)":
            filtered_books.sort(
                key=lambda x: (
                    str(x.get("category") or "Standalone Novel"),
                    -float(x.get("author_fame_score", 0.0)),
                    -int(x.get("sightings_count", 1)),
                    str(x.get("author") or "").lower(),
                    str(x.get("title") or "").lower()
                )
            )
        elif sort_by == "Sightings Count (Most Frequent First)":
            filtered_books.sort(key=lambda x: x.get("sightings_count", 1), reverse=True)
        elif sort_by == "Author Name":
            filtered_books.sort(key=lambda x: str(x.get("author") or "").lower())
        elif sort_by == "Book Title":
            filtered_books.sort(key=lambda x: str(x.get("title") or "").lower())

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unique Titles", f"{len(filtered_books)}")
        m2.metric("Total Sightings", f"{sum(b.get('sightings_count', 1) for b in filtered_books)}")
        m3.metric("Selected Genre", cat_filter)
        m4.metric("Authors In Archive", f"{len(load_author_archive())}")

        st.markdown("---")

        # Enrichment Decision Hub
        if auto_enrich_authors and raw_books:
            cur_a_archive = load_author_archive()
            cur_b_archive = load_book_archive()

            # 1. Unique Authors on shelf
            shelf_authors = sorted(list(set(
                b.get("author", "").strip() for b in raw_books
                if b.get("author") and "unknown" not in b.get("author").lower() and b.get("author").lower() != "author" and not b.get("author").lower().startswith("book ")
            )))
            new_authors = [
                a for a in shelf_authors
                if a.lower() not in cur_a_archive or not cur_a_archive[a.lower()].get("author_fame") or cur_a_archive[a.lower()].get("author_fame") in ["-", "Not publicly reported", ""]
            ]

            # 2. Canonical Book Grouping & Multiple Copies Map
            unique_book_map = {}
            for b in raw_books:
                t_str = (b.get("title") or "").strip()
                a_str = (b.get("author") or "").strip()
                if not t_str or "unidentified" in t_str.lower() or t_str.lower().startswith("book "):
                    continue
                c_key = get_canonical_key(t_str, a_str)
                if c_key not in unique_book_map:
                    unique_book_map[c_key] = []
                unique_book_map[c_key].append(b)

            new_books = [
                k for k in unique_book_map
                if k not in cur_b_archive or not cur_b_archive[k].get("book_sales") or cur_b_archive[k].get("book_sales") in ["-", ""]
            ]
            duplicate_copies_count = max(0, len(raw_books) - len(unique_book_map))

            with st.container():
                st.markdown("### 🌟 Catalog Intelligence Hub")
                st.caption("Zero-cost book & author intelligence via Direct Google AI Studio (`gemini-3.5-flash-lite` • 4,000 RPM Free Tier • $0.00)")

                eh_m1, eh_m2, eh_m3, eh_m4, eh_m5 = st.columns(5)
                eh_m1.metric("Shelf Total", f"{len(raw_books)} Books")
                eh_m2.metric("Unique Titles", f"{len(unique_book_map)}")
                eh_m3.metric("Duplicate Copies", f"{duplicate_copies_count} saved")
                eh_m4.metric("New Authors", f"{len(new_authors)} new")
                eh_m5.metric("Unsearched Books", f"{len(new_books)} unsearched")

                if len(new_authors) == 0 and len(new_books) == 0:
                    st.success("🎉 All shelf books and authors are fully enriched! (100% synced with Google Sheets & local archives)")
                else:
                    opt_col1, opt_col2 = st.columns(2)
                    with opt_col1:
                        st.markdown(
                            f"**🌟 Option 1: Author Lifetime Sales Only**  \n"
                            f"Fetches career sales for the **{len(new_authors)} new authors** not in your Google Sheets archive.  \n"
                            f"• **Est. Cost**: **$0.00** (Google AI Studio Free Tier)  \n"
                            f"• **Est. Speed**: **~1–2s** (10 parallel threads)  \n"
                            f"• **Enriches**: Author Fame & Fame Score for genre ranking"
                        )
                        run_opt1 = st.button(
                            f"🌟 Option 1: Pull Author Sales ({len(new_authors)} New Authors)",
                            key="btn_run_enrich_opt1",
                            disabled=(len(new_authors) == 0),
                            use_container_width=True
                        )

                    with opt_col2:
                        st.markdown(
                            f"**🚀 Option 2: Complete Deep Search (Book + Author)**  \n"
                            f"Deep searches copy sales, TV/film deals, spice rating, protagonist, and author fame for **{len(new_books)} unique books**.  \n"
                            f"• **Deduplication Guaranteed**: {duplicate_copies_count} duplicate shelf copies searched **0 times**; shared authors queried **once**.  \n"
                            f"• **Est. Cost**: **$0.00** (Google AI Studio Free Tier)  \n"
                            f"• **Est. Speed**: **~{max(1, math.ceil(len(new_books)/10))}s** (10 parallel threads)  \n"
                            f"• **Enriches**: Complete book intelligence + author sales"
                        )
                        run_opt2 = st.button(
                            f"🚀 Option 2: Full Deep Search ({len(new_books)} Unique Books)",
                            key="btn_run_enrich_opt2",
                            disabled=(len(new_books) == 0),
                            use_container_width=True
                        )

                    if run_opt1:
                        gem_key = get_gemini_key()
                        if not gem_key:
                            st.error("⚠️ GEMINI_API_KEY is not set in secrets. Please configure it to enable zero-cost search.")
                        else:
                            status_box = st.empty()
                            total_auths = len(new_authors)
                            prog = st.progress(0, text=f"Launching 10 parallel threads across {total_auths} new authors…")
                            completed = 0
                            a_arch = load_author_archive()

                            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                                future_to_author = {
                                    executor.submit(query_direct_gemini_author_fame, a, gem_key, "gemini-3.5-flash-lite"): a
                                    for a in new_authors
                                }
                                for fut in concurrent.futures.as_completed(future_to_author):
                                    a_name = future_to_author[fut]
                                    try:
                                        res = fut.result()
                                        fame = res.get("author_fame", "Not publicly reported")
                                        f_score = compute_author_fame_score(fame, res.get("author_fame_score", 0))
                                        a_arch[a_name.lower()] = {
                                            "author": a_name,
                                            "author_fame": fame,
                                            "author_fame_score": float(f_score),
                                            "evidence": res.get("evidence", "-")
                                        }
                                    except Exception as ex:
                                        a_arch[a_name.lower()] = {
                                            "author": a_name,
                                            "author_fame": "Not publicly reported",
                                            "author_fame_score": 0.0,
                                            "evidence": str(ex)
                                        }
                                    completed += 1
                                    left = total_auths - completed
                                    prog.progress(completed / total_auths, text=f"Author Enrichment [{completed}/{total_auths}] • {left} left: {a_name}")
                                    status_box.info(
                                        f"🌟 **Parallel Author Enrichment (10 Threads)**: **{completed}/{total_auths} complete** ({left} remaining)\n\n"
                                        f"✅ Finished: **{a_name}** ({res.get('latency_ms', '-')} ms) — {res.get('author_fame', '-')}"
                                    )

                            save_author_archive(a_arch)
                            for mb in st.session_state.master_books:
                                a_cur = (mb.get("author") or "").strip().lower()
                                if a_cur in a_arch:
                                    mb["author_fame"] = a_arch[a_cur]["author_fame"]
                                    mb["author_fame_score"] = a_arch[a_cur]["author_fame_score"]
                            save_local_master_catalog(st.session_state.master_books)

                            if is_gsheets_configured():
                                sh_sync, _ = get_gsheet_connection()
                                if sh_sync:
                                    sync_catalog_to_gsheets(sh_sync, st.session_state.master_books, get_canonical_key)
                                    sync_authors_to_gsheets(sh_sync, a_arch)

                            status_box.success(f"🎉 Option 1 Complete! Enriched {total_auths} authors in parallel ($0.00).")
                            time.sleep(1)
                            st.rerun()

                    if run_opt2:
                        gem_key = get_gemini_key()
                        if not gem_key:
                            st.error("⚠️ GEMINI_API_KEY is not set in secrets. Please configure it to enable zero-cost search.")
                        else:
                            status_box = st.empty()
                            # STRICT DEDUPLICATION: query only 1 entry per canonical key
                            items_to_query = [unique_book_map[k][0] for k in new_books]
                            total_items = len(items_to_query)
                            prog = st.progress(0, text=f"Launching 10 parallel threads across {total_items} unique books…")
                            completed = 0
                            b_arch = load_book_archive()
                            a_arch = load_author_archive()

                            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                                future_to_item = {
                                    executor.submit(query_direct_gemini_api, b.get("title", ""), b.get("author", ""), gem_key, "gemini-3.5-flash-lite"): (get_canonical_key(b.get("title", ""), b.get("author", "")), b)
                                    for b in items_to_query
                                }
                                for fut in concurrent.futures.as_completed(future_to_item):
                                    c_key, orig_b = future_to_item[fut]
                                    b_title = orig_b.get("title", "Book")
                                    b_author = orig_b.get("author", "")
                                    try:
                                        res = fut.result()
                                        b_sales = res.get("book_sales", "Not publicly reported")
                                        tv = res.get("tv_deal", "No")
                                        spice = res.get("sensual_rating", "Clean / None")
                                        ev = res.get("evidence", "-")
                                        a_fame = res.get("author_fame", "Not publicly reported")
                                        a_score = compute_author_fame_score(a_fame, res.get("author_fame_score", 0))

                                        # 1. Update book archive
                                        b_arch[c_key] = {
                                            "book_sales": b_sales,
                                            "tv_adaptation": tv,
                                            "sensual_rating": spice,
                                            "evidence": ev
                                        }

                                        # 2. Update author archive if fame discovered
                                        clean_a = b_author.strip().lower()
                                        if clean_a and (clean_a not in a_arch or a_arch[clean_a].get("author_fame") in ["-", "Not publicly reported", ""]):
                                            a_arch[clean_a] = {
                                                "author": b_author.strip(),
                                                "author_fame": a_fame,
                                                "author_fame_score": float(a_score),
                                                "evidence": ev
                                            }

                                        # 3. Update ALL matching copies of this book in master_books simultaneously
                                        for mb in st.session_state.master_books:
                                            mb_key = get_canonical_key(mb.get("title", ""), mb.get("author", ""))
                                            if mb_key == c_key:
                                                mb["sales"] = b_sales
                                                mb["tv_adaptation"] = tv
                                                mb["sensual_romance_flag"] = spice
                                                mb["search_evidence"] = ev
                                                mb["deep_searched"] = True
                                                if a_fame and a_fame != "Not publicly reported":
                                                    mb["author_fame"] = a_fame
                                                    mb["author_fame_score"] = float(a_score)
                                            elif mb.get("author", "").strip().lower() == clean_a:
                                                if a_fame and a_fame != "Not publicly reported":
                                                    mb["author_fame"] = a_fame
                                                    mb["author_fame_score"] = float(a_score)

                                    except Exception as ex:
                                        b_arch[c_key] = {
                                            "book_sales": "Not publicly reported",
                                            "tv_adaptation": "No",
                                            "sensual_rating": "Clean / None",
                                            "evidence": str(ex)
                                        }

                                    completed += 1
                                    left = total_items - completed
                                    prog.progress(completed / total_items, text=f"Deep Search [{completed}/{total_items}] • {left} left: {b_title}")
                                    status_box.info(
                                        f"🚀 **Parallel Deep Search (10 Threads)**: **{completed}/{total_items} complete** ({left} remaining)\n\n"
                                        f"✅ Finished: **{b_title}** ({res.get('latency_ms', '-')} ms) — Sales: {res.get('book_sales', '-')} | Author: {res.get('author_fame', '-')}"
                                    )

                            save_book_archive(b_arch)
                            save_author_archive(a_arch)
                            save_local_master_catalog(st.session_state.master_books)

                            if is_gsheets_configured():
                                sh_sync, _ = get_gsheet_connection()
                                if sh_sync:
                                    sync_catalog_to_gsheets(sh_sync, st.session_state.master_books, get_canonical_key)
                                    sync_authors_to_gsheets(sh_sync, a_arch)
                                    sync_books_to_gsheets(sh_sync, b_arch)

                            status_box.success(f"🎉 Option 2 Complete! Deep searched {total_items} unique books and updated all shelf copies ($0.00).")
                            time.sleep(1)
                            st.rerun()

                st.markdown("---")

        img_col, table_col = st.columns([1, 1.2])

        with img_col:
            st.subheader("📷 Shelf Highlights")
            if st.session_state.processed_images:
                img_choice = st.selectbox("Select Image to Inspect", list(st.session_state.processed_images.keys()))
                selected_img = st.session_state.processed_images[img_choice]
                tab_zoom, tab_static = st.tabs(["🔍 Interactive Zoom & Tap", "🖼️ Overview"])

                image_books = [
                    b for b in st.session_state.master_books
                    if b.get("image_name") == img_choice or not b.get("image_name")
                ]
                if not image_books:
                    image_books = st.session_state.master_books

                with tab_zoom:
                    if _shelf_inspector is not None:
                        buf = io.BytesIO()
                        selected_img.save(buf, format="JPEG", quality=88)
                        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

                        # Build clean, strictly JSON-serializable list of books (no numpy ndarrays)
                        clean_books = []
                        img_w, img_h = selected_img.size
                        for b in image_books:
                            box = b.get("box_2d")
                            if not box and b.get("box_pixels"):
                                px = b.get("box_pixels")
                                box = [
                                    int((px[1] / float(img_h)) * 1000),
                                    int((px[0] / float(img_w)) * 1000),
                                    int((px[3] / float(img_h)) * 1000),
                                    int((px[2] / float(img_w)) * 1000),
                                ]
                            elif box and isinstance(box, np.ndarray):
                                box = box.tolist()
                            elif box:
                                box = [int(v) for v in box]
                            else:
                                box = [0, 0, 0, 0]

                            clean_books.append({
                                "id": int(b.get("id", 0)),
                                "box_2d": box,
                                "title": str(b.get("title") or ""),
                                "author": str(b.get("author") or ""),
                                "category": str(b.get("category") or "General Fiction"),
                                "series": str(b.get("series") or "-"),
                                "protagonist": str(b.get("protagonist") or "-"),
                                "author_fame": str(b.get("author_fame") or "-"),
                                "sales": str(b.get("sales") or "-"),
                                "tv_adaptation": str(b.get("tv_adaptation") or "-"),
                                "sensual_romance_flag": str(b.get("sensual_romance_flag") or "-"),
                                "deep_searched": bool(b.get("deep_searched", False)),
                            })

                        sel_id = st.session_state.get("selected_book_id")
                        sel_id_val = int(sel_id) if sel_id is not None else None

                        if "handled_inspector_reqs" not in st.session_state:
                            st.session_state.handled_inspector_reqs = set()

                        inspector_event = _shelf_inspector(
                            image_b64=b64_data,
                            books=clean_books,
                            selected_id=sel_id_val,
                            key=f"shelf_insp_{img_choice}"
                        )
                        if inspector_event and isinstance(inspector_event, dict):
                            act = inspector_event.get("action")
                            b_id = inspector_event.get("book_id")
                            if act == "select":
                                st.session_state["selected_book_id"] = b_id
                            elif act == "deep_search":
                                req_id = inspector_event.get("req_id") or f"{b_id}_{inspector_event.get('title')}"
                                # HARD GUARD: Never process the same event twice across st.rerun()!
                                if req_id not in st.session_state.handled_inspector_reqs:
                                    st.session_state.handled_inspector_reqs.add(req_id)
                                    t = inspector_event.get("title")
                                    a = inspector_event.get("author")
                                    if t:
                                        with st.spinner(f"🌐 Deep searching '{t}' by {a} ($0.00 via Gemini 3.5 Flash-Lite)…"):
                                            res = deep_search_single_book(t, a)
                                            for mb in st.session_state.master_books:
                                                if get_canonical_key(mb.get("title", ""), mb.get("author", "")) == get_canonical_key(t, a):
                                                    mb["sales"] = res.get("book_sales", "Not publicly reported")
                                                    mb["tv_adaptation"] = res.get("tv_adaptation", "No")
                                                    mb["sensual_romance_flag"] = res.get("sensual_rating", "Clean / None")
                                                    mb["search_evidence"] = res.get("evidence", "-")
                                                    mb["deep_searched"] = True
                                                    if res.get("author_fame") and res.get("author_fame") != "Not publicly reported":
                                                        mb["author_fame"] = res.get("author_fame")
                                                        mb["author_fame_score"] = float(res.get("author_fame_score", 0.0) or 0.0)
                                            save_local_master_catalog(st.session_state.master_books)
                                            st.session_state["selected_book_id"] = b_id
                                            st.rerun()
                    else:
                        render_zoomable_image(selected_img, height=620)
                with tab_static:
                    st.image(selected_img, caption=img_choice, width="stretch")

        with table_col:
            st.subheader(f"📋 Master Catalog ({len(filtered_books)} Unique Titles)")

            # On-Demand Single Book Deep Dive UI
            with st.expander("🔍 **Deep Dive Into a Book** (On-Demand Deep Search: $0.00)", expanded=True):
                st.caption("Inspect exact print/ebook/audiobook sales, TV/movie adaptation deals, and spice/romance ratings for an individual book ($0.00 via Gemini 3.5 Flash-Lite).")

                # Strictly sort numerically by ID and Shelf so users can lookup by number
                sorted_by_id = sorted(
                    filtered_books,
                    key=lambda b: (int(b.get("shelf", 1) or 1), int(b.get("id", 0) or 0))
                )
                book_options = {
                    f"#{b.get('id', i+1)}: {b.get('title')} — {b.get('author')} (Shelf {b.get('shelf', 1)})": b
                    for i, b in enumerate(sorted_by_id)
                }

                if book_options:
                    default_idx = 0
                    sel_id = st.session_state.get("selected_book_id")
                    if sel_id is not None:
                        for idx, b in enumerate(sorted_by_id):
                            if b.get("id") == sel_id:
                                default_idx = idx
                                break

                    dive_c1, dive_c2 = st.columns([3, 1])
                    with dive_c1:
                        selected_label = st.selectbox(
                            "Select book by # (ordered numerically):", 
                            list(book_options.keys()),
                            index=default_idx
                        )
                        if selected_label:
                            st.session_state["selected_book_id"] = book_options[selected_label].get("id")
                    with dive_c2:
                        st.write("")
                        st.write("")
                        run_deep_dive = st.button("🚀 Deep Search Book", key="btn_deep_dive")

                    if selected_label and selected_label in book_options:
                        cur_b = book_options[selected_label]
                        st.info(
                            f"**#{cur_b.get('id')}: {cur_b.get('title')}** by *{cur_b.get('author')}*  \n"
                            f"📁 **Category**: `{cur_b.get('category', 'Standalone')}` | **Series**: `{cur_b.get('series', '-')}`  \n"
                            f"🌟 **Author Career Sales**: {cur_b.get('author_fame', '-')}  \n"
                            f"📖 **Book Sales / Listens**: {cur_b.get('sales', '-')}  \n"
                            f"📺 **TV / Film Deal**: {cur_b.get('tv_adaptation', '-')}  \n"
                            f"💘 **Romance Rating**: {cur_b.get('sensual_romance_flag', '-')}"
                        )

                    if run_deep_dive and selected_label:
                        target_book = book_options[selected_label]
                        t = target_book.get("title")
                        a = target_book.get("author")
                        gem_k = get_gemini_key()
                        if not gem_k:
                            st.warning("⚠️ GEMINI_API_KEY is not set in secrets.")
                        else:
                            with st.spinner(f"🌐 Querying Gemini 3.5 Flash-Lite for '{t}' by {a} ($0.00)…"):
                                res = deep_search_single_book(t, a)
                                for mb in st.session_state.master_books:
                                    if get_canonical_key(mb.get("title", ""), mb.get("author", "")) == get_canonical_key(t, a):
                                        mb["sales"] = res.get("book_sales", "Not publicly reported")
                                        mb["tv_adaptation"] = res.get("tv_adaptation", "No")
                                        mb["sensual_romance_flag"] = res.get("sensual_rating", "Clean / None")
                                        mb["search_evidence"] = res.get("evidence", "-")
                                        mb["deep_searched"] = True
                                        if res.get("author_fame") and res.get("author_fame") != "Not publicly reported":
                                            mb["author_fame"] = res.get("author_fame")
                                            mb["author_fame_score"] = float(res.get("author_fame_score", 0.0) or 0.0)
                                save_local_master_catalog(st.session_state.master_books)
                                st.success(f"✅ Deep search complete for **{t}**!")
                                st.rerun()

            table_rows = []
            for b in filtered_books:
                table_rows.append({
                    "ID": b.get("id"),
                    "Title": b.get("title"),
                    "Author": b.get("author"),
                    "Shelf": b.get("shelf", 1),
                    "Author Career Sales": b.get("author_fame", "-"),
                    "Book Sales / Listens": b.get("sales", "-"),
                    "TV / Film Deal": b.get("tv_adaptation", "-"),
                    "Romance Rating": b.get("sensual_romance_flag", "-"),
                    "Genre / Category": b.get("category", "Standalone Novel"),
                    "Series / Protagonist": f"{b.get('series')} ({b.get('protagonist')})" if b.get("protagonist") != "-" else b.get("series"),
                    "Sightings": f"{b.get('sightings_count')}x",
                    "Locations": ", ".join(b.get("all_locations", [])),
                    "Search Evidence": b.get("search_evidence", "-"),
                    "Model": b.get("source", "API")
                })
            st.dataframe(table_rows, width="stretch", height=560)

            # Export / Download Buttons
            dl_col1, dl_col2 = st.columns([1, 1])
            with dl_col1:
                cat_json_str = json.dumps(table_rows, indent=2, ensure_ascii=False)
                st.download_button(
                    "📥 Download Catalog (JSON)",
                    data=cat_json_str,
                    file_name="master_bookshelf_catalog.json",
                    mime="application/json",
                    use_container_width=True
                )
            with dl_col2:
                if table_rows:
                    import csv
                    csv_io = io.StringIO()
                    csv_writer = csv.DictWriter(csv_io, fieldnames=list(table_rows[0].keys()))
                    csv_writer.writeheader()
                    csv_writer.writerows(table_rows)
                    st.download_button(
                        "📥 Download Catalog (CSV / Excel)",
                        data=csv_io.getvalue(),
                        file_name="master_bookshelf_catalog.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


with tab_arena:
    st.header("⚔️ Gemini 3.8 Flash Vision Arena: OpenRouter vs Direct Google AI Studio")
    st.markdown(
        "Compare the exact same bookshelf shelf photo processed by **OpenRouter (`google/gemini-3.8-flash`)** vs "
        "**Direct Google AI Studio (`gemini-3.8-flash`)** on your Tier 1 Pay-As-You-Go account before retiring your OpenRouter key."
    )

    # Tier 1 Quota & Cost Breakdown Card
    with st.expander("ℹ️ **Tier 1 Pay-As-You-Go Limits & Cost Breakdown (Click to expand)**", expanded=True):
        q1, q2 = st.columns(2)
        with q1:
            st.markdown(
                "#### 🟢 Direct Google AI Studio (Tier 1 Pay-As-You-Go)\n"
                "• **RPM (Requests / Min)**: `1,000 RPM` (up to `4,000 RPM` on Flash-Lite)\n"
                "• **TPM (Tokens / Min)**: `2,000,000 TPM` (~900 full shelf photos/min)\n"
                "• **RPD (Requests / Day)**: `10,000 RPD`\n"
                "• **Pricing**: **$0.075** / 1M input tokens • **$0.30** / 1M output tokens\n"
                "• **Per Shelf Photo**: **~$0.0006 - $0.0008** *(Direct pipe, lowest latency)*"
            )
        with q2:
            st.markdown(
                "#### 🟠 OpenRouter Proxy (Current Shelf Vision)\n"
                "• **RPM / TPM**: Shared proxy queue; variable rate limits\n"
                "• **RPD**: Uncapped subject to prepaid balance\n"
                "• **Pricing**: **$0.75** / 1M input tokens • **$3.75** / 1M output tokens\n"
                "• **Per Shelf Photo**: **~$0.0075** *(~12.5x more expensive due to proxy markup)*\n"
                "• **Latency**: Extra intermediate routing hops"
            )

    # API Keys Configuration
    openrouter_k = get_openrouter_key()
    gemini_k = get_gemini_key()

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if openrouter_k:
            masked_or = (openrouter_k[:6] + "…" + openrouter_k[-4:]) if len(openrouter_k) > 10 else "••••••••"
            st.success(f"🟢 **OpenRouter Key**: `{masked_or}`")
        else:
            openrouter_k = st.text_input("Enter OpenRouter Key:", type="password", key="arena_or_key")

    with col_k2:
        if gemini_k:
            masked_gem = (gemini_k[:6] + "…" + gemini_k[-4:]) if len(gemini_k) > 10 else "••••••••"
            st.success(f"🟢 **Google AI Studio Key**: `{masked_gem}` (Tier 1)")
        else:
            gemini_k = st.text_input("Enter Gemini Key:", type="password", key="arena_gem_key")

    st.markdown("---")
    st.subheader("🖼️ Select Test Shelf Photo")

    test_source = st.radio(
        "Choose photo source for benchmark:",
        ["📚 Pre-bundled Example Bookstore Shelf (`data/sample_shelf.jpg`)", "📤 Upload Custom Shelf Photo (Phone / Pixel / Camera)"],
        horizontal=True
    )

    arena_test_bgr = None
    arena_img_label = "sample_shelf.jpg"

    if "Pre-bundled" in test_source:
        if os.path.exists(SAMPLE_IMAGE):
            with open(SAMPLE_IMAGE, "rb") as f:
                img_bytes = f.read()
            arena_test_bgr, _ = decode_photo(img_bytes)
            arena_img_label = "sample_shelf.jpg"
            if arena_test_bgr is not None:
                h_i, w_i = arena_test_bgr.shape[:2]
                st.caption(f"Loaded `{SAMPLE_IMAGE}` ({w_i}x{h_i} px, {len(img_bytes)//1024} KB)")
        else:
            st.error(f"Sample image not found at `{SAMPLE_IMAGE}`")
    else:
        up_bench = st.file_uploader("Upload shelf photo for arena comparison", type=UPLOAD_TYPES, key="arena_shelf_upload")
        if up_bench is not None:
            up_bytes = up_bench.getvalue()
            arena_test_bgr, _ = decode_photo(up_bytes)
            arena_img_label = up_bench.name
            if arena_test_bgr is not None:
                h_i, w_i = arena_test_bgr.shape[:2]
                st.caption(f"Uploaded `{arena_img_label}` ({w_i}x{h_i} px, {len(up_bytes)//1024} KB)")

    if arena_test_bgr is not None:
        with st.expander("🔍 Preview Benchmark Photo", expanded=False):
            st.image(cv2.cvtColor(arena_test_bgr, cv2.COLOR_BGR2RGB), width=450)

    # Action Buttons
    st.markdown("---")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        run_or = st.button("⚡ Test OpenRouter (gemini-3.8-flash)", key="btn_run_or_vision", use_container_width=True, disabled=not bool(openrouter_k and arena_test_bgr is not None))
    with b_col2:
        run_gem = st.button("🚀 Test Direct AI Studio (gemini-3.8-flash)", key="btn_run_gem_vision", use_container_width=True, disabled=not bool(gemini_k and arena_test_bgr is not None))
    with b_col3:
        run_h2h = st.button("⚔️ Run Simultaneous Head-to-Head", key="btn_run_h2h_vision", use_container_width=True, disabled=not bool(openrouter_k and gemini_k and arena_test_bgr is not None))

    if "vision_arena_results" not in st.session_state:
        st.session_state.vision_arena_results = None

    if run_or and openrouter_k and arena_test_bgr is not None:
        with st.spinner("⚡ Sending photo to OpenRouter (gemini-3.8-flash)…"):
            res = run_vision_benchmark_openrouter(arena_test_bgr, openrouter_k, model_id="google/gemini-3.8-flash")
            st.session_state.vision_arena_results = {
                "mode": "OpenRouter Vision",
                "photo": arena_img_label,
                "data": [res]
            }
            st.rerun()

    if run_gem and gemini_k and arena_test_bgr is not None:
        with st.spinner("🚀 Sending photo directly to Google AI Studio (gemini-3.8-flash)…"):
            res = run_vision_benchmark_direct_gemini(arena_test_bgr, gemini_k, model_id="gemini-3.8-flash")
            st.session_state.vision_arena_results = {
                "mode": "Direct Google AI Studio Vision",
                "photo": arena_img_label,
                "data": [res]
            }
            st.rerun()

    if run_h2h and openrouter_k and gemini_k and arena_test_bgr is not None:
        with st.spinner("⚔️ Running parallel head-to-head comparison on exact same shelf image…"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f_gem = executor.submit(run_vision_benchmark_direct_gemini, arena_test_bgr, gemini_k, "gemini-3.8-flash")
                f_or = executor.submit(run_vision_benchmark_openrouter, arena_test_bgr, openrouter_k, "google/gemini-3.8-flash")
                res_gem = f_gem.result()
                res_or = f_or.result()
            st.session_state.vision_arena_results = {
                "mode": "Head-to-Head Comparison",
                "photo": arena_img_label,
                "data": [res_gem, res_or]
            }
            st.rerun()

    # Results Display
    if st.session_state.vision_arena_results:
        v_res = st.session_state.vision_arena_results
        st.markdown("---")
        st.subheader(f"📊 Results: {v_res['mode']} (`{v_res.get('photo', 'shelf.jpg')}`)")

        data_rows = v_res["data"]
        
        # Display side-by-side metric cards
        if len(data_rows) == 2:
            gem_r = data_rows[0]
            or_r = data_rows[1]
            c_g, c_o = st.columns(2)
            with c_g:
                st.markdown(f"### 🚀 {gem_r['platform']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{gem_r['latency_ms']} ms", delta=f"{round(or_r['latency_ms'] - gem_r['latency_ms'], 1)} ms faster" if or_r['latency_ms'] > gem_r['latency_ms'] else None)
                m2.metric("Books Found", gem_r["books_count"])
                m3.metric("Cost", f"${gem_r['cost_usd']:.6f}")
                st.caption(f"Tokens: {gem_r.get('in_tokens', '-')} in / {gem_r.get('out_tokens', '-')} out • Status: `{gem_r['status']}`")

            with c_o:
                st.markdown(f"### ⚡ {or_r['platform']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{or_r['latency_ms']} ms")
                m2.metric("Books Found", or_r["books_count"])
                m3.metric("Cost", f"${or_r['cost_usd']:.5f}")
                st.caption(f"Tokens: {or_r.get('in_tokens', '-')} in / {or_r.get('out_tokens', '-')} out • Status: `{or_r['status']}`")
        elif len(data_rows) == 1:
            single_r = data_rows[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Platform", single_r["platform"])
            m2.metric("Latency", f"{single_r['latency_ms']} ms")
            m3.metric("Books Found", single_r["books_count"])
            m4.metric("Cost", f"${single_r['cost_usd']:.6f}")

        # Summary Table
        table_records = []
        for r in data_rows:
            table_records.append({
                "Platform": r.get("platform"),
                "Model": r.get("model"),
                "Latency (ms)": r.get("latency_ms"),
                "Books Found": r.get("books_count"),
                "In Tokens": r.get("in_tokens", 0),
                "Out Tokens": r.get("out_tokens", 0),
                "Cost (USD)": f"${r.get('cost_usd', 0.0):.6f}",
                "Status": r.get("status")
            })
        st.dataframe(table_records, width="stretch")

        # Side-by-side Visual Shelf Comparison
        if arena_test_bgr is not None:
            st.markdown("### 🖼️ Side-by-Side Visual Shelf Detection Overlay")
            if len(data_rows) == 2:
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    st.markdown(f"#### 🚀 {data_rows[0]['platform']} ({data_rows[0]['books_count']} books)")
                    ann_img1 = draw_annotated_vision_result(arena_test_bgr, data_rows[0].get("books", []))
                    if ann_img1:
                        st.image(ann_img1, caption=f"{data_rows[0]['platform']}: {data_rows[0]['books_count']} books detected", use_container_width=True)
                with img_c2:
                    st.markdown(f"#### ⚡ {data_rows[1]['platform']} ({data_rows[1]['books_count']} books)")
                    ann_img2 = draw_annotated_vision_result(arena_test_bgr, data_rows[1].get("books", []))
                    if ann_img2:
                        st.image(ann_img2, caption=f"{data_rows[1]['platform']}: {data_rows[1]['books_count']} books detected", use_container_width=True)
            elif len(data_rows) == 1:
                st.markdown(f"#### 📸 {data_rows[0]['platform']} ({data_rows[0]['books_count']} books)")
                ann_single = draw_annotated_vision_result(arena_test_bgr, data_rows[0].get("books", []))
                if ann_single:
                    st.image(ann_single, caption=f"{data_rows[0]['platform']}: {data_rows[0]['books_count']} books detected", use_container_width=True)

        # 1-Click Copyable Output for Chat (CRITICAL)
        st.markdown("### 📋 Copyable Output (Click Copy button in top-right corner to paste in chat)")
        headers = ["Platform", "Model", "Latency (ms)", "Books Found", "In Tokens", "Out Tokens", "Cost (USD)", "Status"]
        md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in table_records:
            md_lines.append("| " + " | ".join(str(row.get(h, "-")).replace("\n", " ").replace("|", "/") for h in headers) + " |")
        raw_markdown = "\n".join(md_lines)
        st.code(raw_markdown, language="markdown")

        with st.expander("🔍 View Raw JSON"):
            st.code(json.dumps(data_rows, indent=2, ensure_ascii=False), language="json")

        # Books list inspection
        with st.expander("📚 Inspect Detected Books"):
            for r in data_rows:
                st.markdown(f"**{r.get('platform')} ({len(r.get('books', []))} books)**")
                b_preview = [{"Shelf": b.get("shelf_row", 1), "Title": b.get("title", "-"), "Author": b.get("author", "-"), "Spine": b.get("spine_text", "-")} for b in r.get("books", [])]
                st.dataframe(b_preview, width="stretch")


st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Bookshelf AI • High Speed Vision")
