import streamlit as st
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
from PIL import Image, ImageOps

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

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_UPLOAD_DIM = 1024       # px on the long edge sent to the model
JPEG_QUALITY = 80
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
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

# ---------------------------------------------------------------------------
# OpenRouter vision API
#
# Everything here streams. A shelf photo can legitimately take 30-90s to
# describe -- a single blocking request gives the UI nothing to show for that
# whole time, which is indistinguishable from a hang. Streaming lets the caller
# report bytes-arriving and seconds-elapsed while the model writes.
# ---------------------------------------------------------------------------

VISION_PROMPT = """Analyze this bookstore bookshelf image. Detect every book visible across all shelves and bookcases.
For each book, identify its bounding box and metadata.
Return a valid JSON object:
{
  "books": [
    {
      "box_2d": [ymin, xmin, ymax, xmax],
      "shelf_row": 1,
      "title": "Canonical Title",
      "author": "Author Name",
      "category": "Strict Sequential Series" | "Recurring Protagonist" | "Standalone Novel",
      "series_info": "Series Name #Number or -",
      "protagonist": "Lead Character or -",
      "sensual_flag": "❌ Explicit Romance / Sensual" | "⚠️ Sensual Infidelity Elements" | "✔️ None (Pure Thriller / Mystery)",
      "tv_adaptation": "📺 Yes (Show Title / Network)" | "🎬 Optioned / In Prod." | "❌ No",
      "sales_popularity": "Estimated bestseller level or Standard"
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
                piece = (choice.get("delta") or {}).get("content") or ""
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
        "max_tokens": 1,
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

# Helper to retrieve API key securely
def get_openrouter_key():
    try:
        if hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    if "OPENROUTER_API_KEY" in os.environ:
        return os.environ["OPENROUTER_API_KEY"]
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        return line.strip().split("=", 1)[1].strip("\"'")
        except Exception:
            pass
    return None

detected_key = get_openrouter_key()

# Sidebar: API & Model Selection
st.sidebar.header("⚡ Model & Speed Controls")

if detected_key:
    st.sidebar.success("✅ OpenRouter Key Connected")
    api_key = detected_key
else:
    api_key = st.sidebar.text_input("Enter OpenRouter API Key", type="password")

selected_model = st.sidebar.selectbox(
    "🚀 AI Vision Model (Speed vs Detail)",
    [
        "google/gemini-2.5-flash-lite",  # Ultra-fast (~1.5s)
        "google/gemini-2.5-flash",       # Standard (~3.5s)
        "openai/gpt-4o-mini"             # Fast alternative (~2s)
    ],
    index=0,
    help="Gemini 2.5 Flash-Lite is 3x faster with sub-2s response times and lower token cost."
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

if st.sidebar.button("🗑️ Reset / Clear All"):
    st.session_state.processed_images = {}
    st.session_state.master_books = []
    st.session_state.pending_uploads = {}
    # Bumping the nonce rebuilds the uploader widget, which is the only way to
    # drop files it is already holding.
    st.session_state.uploader_nonce += 1
    st.session_state.upload_counter += 1
    st.rerun()

if st.sidebar.button("🔌 Test API Connection"):
    if not api_key:
        st.sidebar.error("No API key set.")
    else:
        ok, msg = ping_api(selected_model, api_key)
        (st.sidebar.success if ok else st.sidebar.error)(msg)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filters")

sensual_filter = st.sidebar.selectbox(
    "💘 Romantic / Sensual Content",
    ["All Books", "✔️ Clean Only (No Explicit Romance)", "❌ Explicit Romance / Sensual Only"]
)

tv_filter = st.sidebar.selectbox(
    "📺 TV / Screen Adaptation",
    ["All Books", "📺 TV / Screen Adapted Only", "❌ Non-Adapted Only"]
)

cat_filter = st.sidebar.selectbox(
    "📖 Story Category", 
    ["All Categories", "Strict Sequential Series", "Recurring Protagonist", "Standalone Novel"]
)

sort_by = st.sidebar.selectbox(
    "📊 Sort Master Catalog By", 
    ["Most Sales / Popularity", "Sightings Count (Most Frequent First)", "Author Name", "Book Title"]
)

# Process Image
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


def process_bookshelf(img_bytes, image_id, image_name, mode, model_id, key, status_cb=None):
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
        api_books = call_vision_api(img, model_id, key, status_cb)
        for idx, ab in enumerate(api_books, start=1):
            ymin, xmin, ymax, xmax = ab.get("box_2d", [0, 0, 0, 0])
            px_ymin = int((ymin / 1000.0) * H)
            px_xmin = int((xmin / 1000.0) * W)
            px_ymax = int((ymax / 1000.0) * H)
            px_xmax = int((xmax / 1000.0) * W)
            
            books_out.append({
                "id": idx,
                "image_id": image_id,
                "image_name": image_name,
                "shelf": ab.get("shelf_row", 1),
                "title": ab.get("title", f"Book {idx}"),
                "author": ab.get("author", "Unknown"),
                "category": ab.get("category", "Standalone Novel"),
                "series": ab.get("series_info", "-"),
                "protagonist": ab.get("protagonist", "-"),
                "sensual_romance_flag": ab.get("sensual_flag", "✔️ None (Pure Thriller / Mystery)"),
                "tv_adaptation": ab.get("tv_adaptation", "❌ No"),
                "sales": ab.get("sales_popularity", "Standard"),
                "sales_score": 10.0 if "Bestseller" in ab.get("sales_popularity", "") else 1.0,
                "box_pixels": [px_xmin, px_ymin, px_xmax, px_ymax],
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
                    "sensual_romance_flag": "✔️ None (Pure Thriller / Mystery)",
                    "tv_adaptation": "❌ No",
                    "sales": "Standard",
                    "sales_score": 1.0,
                    "box_pixels": [xmin, ymin, xmax, ymax],
                    "source": "Offline OCR"
                })

    # Render Annotated Overlay
    annotated = img.copy()
    overlay = img.copy()
    colors = [(50, 220, 100), (240, 150, 40), (220, 60, 220), (30, 200, 240), (255, 100, 50), (100, 100, 255)]
    
    for b in books_out:
        xmin, ymin, xmax, ymax = b["box_pixels"]
        col = colors[(b["shelf"] - 1) % len(colors)]
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), col, -1)
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), col, 2)
        
        badge_y = max(14, ymin)
        cv2.circle(annotated, (xmin + 8, badge_y), 8, (10, 10, 10), -1)
        cv2.putText(annotated, str(b["id"]), (xmin + 4 if b["id"] < 10 else xmin + 1, badge_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
                    
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
    pil_res = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    return pil_res, books_out, None

def get_canonical_key(title, author):
    clean_t = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
    clean_a = re.sub(r'[^a-zA-Z0-9]', '', author.lower())
    return f"{clean_t}_{clean_a}"

# Main Upload Area
st.markdown("### 📸 Select Bookshelf Photos")

upload_tab1, upload_tab2, upload_tab3 = st.tabs([
    "📱 Phone Gallery (Stack Photos)",
    "📸 Live Camera",
    "💻 Desktop / Batch Upload"
])

with upload_tab1:
    st.caption("Pick photos one by one from your phone gallery. They stack in the ready queue below.")
    picker_label = "➕ Tap to pick photo from gallery" if not st.session_state.pending_uploads else "➕ Tap to add another shelf photo"
    new_mobile_file = st.file_uploader(
        picker_label,
        type=UPLOAD_TYPES,
        accept_multiple_files=False,
        key=f"shelf_picker_{st.session_state.upload_counter}",
        help="Select a bookshelf photo to add to your scan queue.",
    )
    if new_mobile_file is not None:
        file_bytes = new_mobile_file.getvalue()
        if file_bytes:
            file_key = f"{new_mobile_file.name}:{len(file_bytes)}"
            if file_key not in st.session_state.pending_uploads:
                st.session_state.pending_uploads[file_key] = (new_mobile_file.name, file_bytes)
                st.session_state.upload_counter += 1
                st.rerun()

with upload_tab2:
    st.caption("Snap a photo of your bookshelf directly using your phone's camera.")
    camera_photo = st.camera_input("Take shelf photo", key="shelf_camera_input")
    if camera_photo is not None:
        cam_bytes = camera_photo.getvalue()
        if cam_bytes:
            cam_name = f"Camera_Shelf_{len(st.session_state.pending_uploads) + 1}.jpg"
            cam_key = f"{cam_name}:{len(cam_bytes)}"
            if cam_key not in st.session_state.pending_uploads:
                st.session_state.pending_uploads[cam_key] = (cam_name, cam_bytes)
                st.rerun()

with upload_tab3:
    st.caption("Select multiple shelf photos at once (best for desktop browsers or folders).")
    batch_files = st.file_uploader(
        "Choose multiple bookshelf photos",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"batch_uploader_{st.session_state.uploader_nonce}",
    )
    if batch_files:
        for f in batch_files:
            b_key = f"{f.name}:{f.size}"
            if b_key not in st.session_state.pending_uploads:
                st.session_state.pending_uploads[b_key] = (f.name, f.getvalue())

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
                del st.session_state.pending_uploads[k]
                st.rerun()

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
            pil_img, books, err = process_bookshelf(
                img_bytes, idx, f"Image {idx}", scanner_mode,
                selected_model, api_key, status_cb,
            )
            if err:
                failures.append(f"**{name}** — {err}")
            elif pil_img is not None:
                st.session_state.processed_images[f"Image {idx} ({name})"] = pil_img
                st.session_state.master_books.extend(books)
            progress.progress(idx / len(queued))

        status.empty()
        progress.empty()
        for failure in failures:
            st.error(f"❌ {failure}")
        st.success(
            f"🎉 Finished in {time.time() - run_started:.1f}s — "
            f"{len(st.session_state.master_books)} books across "
            f"{len(queued) - len(failures)} of {len(queued)} photo(s)."
        )

# Pre-bundled demo button
st.markdown("---")
if st.button("🧪 Test with Example Shelf (Pre-bundled)", width="stretch"):
    if os.path.exists(SAMPLE_IMAGE):
        with open(SAMPLE_IMAGE, "rb") as f:
            demo_bytes = f.read()
        status = st.empty()
        pil_img, books, err = process_bookshelf(
            demo_bytes, 1, "Image 1 (Example Bookstore Shelf)", scanner_mode,
            selected_model, api_key,
            lambda msg: status.info(f"**Example shelf**\n\n{msg}"),
        )
        status.empty()
        if err:
            st.error(f"❌ Example shelf — {err}")
        else:
            st.session_state.processed_images = {"Image 1 (Example Bookstore Shelf)": pil_img}
            st.session_state.master_books = books
            st.success(f"🎉 Example shelf loaded: {len(books)} books identified!")
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
        flag = b.get("sensual_romance_flag", "")
        if sensual_filter == "✔️ Clean Only (No Explicit Romance)" and "❌" in flag:
            continue
        if sensual_filter == "❌ Explicit Romance / Sensual Only" and "❌" not in flag:
            continue
            
        tv = b.get("tv_adaptation", "")
        if tv_filter == "📺 TV / Screen Adapted Only" and not tv.startswith("📺"):
            continue
        if tv_filter == "❌ Non-Adapted Only" and tv.startswith("📺"):
            continue
            
        if cat_filter != "All Categories" and b.get("category", "") != cat_filter:
            continue
            
        filtered_books.append(b)

    # Sort
    if sort_by == "Most Sales / Popularity":
        filtered_books.sort(key=lambda x: x.get("sales_score", 0.0), reverse=True)
    elif sort_by == "Sightings Count (Most Frequent First)":
        filtered_books.sort(key=lambda x: x.get("sightings_count", 1), reverse=True)
    elif sort_by == "Author Name":
        filtered_books.sort(key=lambda x: x.get("author", ""))
    elif sort_by == "Book Title":
        filtered_books.sort(key=lambda x: x.get("title", ""))

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique Titles", f"{len(filtered_books)}")
    m2.metric("Total Sightings", f"{sum(b.get('sightings_count', 1) for b in filtered_books)}")
    m3.metric("TV Adapted", f"{sum(1 for b in filtered_books if b.get('tv_adaptation','').startswith('📺'))}")
    m4.metric("Non-Adapted", f"{sum(1 for b in filtered_books if not b.get('tv_adaptation','').startswith('📺'))}")

    st.markdown("---")

    img_col, table_col = st.columns([1, 1.2])

    with img_col:
        st.subheader("📷 Shelf Highlights")
        if st.session_state.processed_images:
            img_choice = st.selectbox("Select Image to Inspect", list(st.session_state.processed_images.keys()))
            selected_img = st.session_state.processed_images[img_choice]
            st.image(selected_img, caption=img_choice, width="stretch")

    with table_col:
        st.subheader(f"📋 Master Catalog ({len(filtered_books)} Unique Titles)")
        table_rows = []
        for b in filtered_books:
            table_rows.append({
                "Sightings": f"{b.get('sightings_count')}x",
                "Locations": ", ".join(b.get("all_locations", [])),
                "Title": b.get("title"),
                "Author": b.get("author"),
                "Category": b.get("category"),
                "Series / Protagonist": f"{b.get('series')} ({b.get('protagonist')})" if b.get("protagonist") != "-" else b.get("series"),
                "Romance Flag": b.get("sensual_romance_flag", "✔️ None"),
                "TV Adaptation": b.get("tv_adaptation", "❌ No"),
                "Sales Rank": b.get("sales", "Standard"),
                "Model": b.get("source", "API")
            })
        st.dataframe(table_rows, width="stretch", height=620)

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Bookshelf AI • High Speed Vision")
