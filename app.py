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

_CLIENT_UPLOADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_uploader")
if os.path.exists(_CLIENT_UPLOADER_DIR):
    _client_uploader = components.declare_component("fast_shelf_uploader", path=_CLIENT_UPLOADER_DIR)
else:
    _client_uploader = None

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
MAX_UPLOAD_DIM = 4096       # 4K px on the long edge sent to the model (~4-5MB payload)
JPEG_QUALITY = 92
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
if "arena_results" not in st.session_state:
    st.session_state.arena_results = {}

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

CRITICAL GROUNDING & ACCURACY RULES:
1. GROUNDING & VISIBLE TEXT: For each book, you MUST first read the exact visible text printed on that specific spine ("spine_text"). The "title" and "author" MUST strictly correspond to that spine_text.
2. DUPLICATE COPIES: Bookstores frequently shelve 2 or more identical copies of the same book side-by-side (e.g. multiple copies of "The Maidens"). You MUST create a separate entry for EVERY physical spine with its own bounding box. NEVER collapse or skip duplicate copies.
3. ANTI-HALLUCINATION: If a spine is too dark, thin, or blurry to read, set "spine_text": "Unreadable", "title": "Unidentified Book", "author": "Unknown". NEVER invent or hallucinate authors or titles (such as James Patterson) for books you cannot clearly read.
4. ORDER & TILT: Order entries shelf by shelf, and left to right (increasing xmin). Estimate "tilt_angle" in degrees from vertical (-30 to +30, 0 = upright, negative = leaning left, positive = leaning right).
5. MAIN SHELF ONLY: Catalog ONLY books standing upright on the primary shelf of this image. Completely ignore cut-off book tops, bottoms, or partial slivers peeking in across the top or bottom frame borders.

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
    "🚀 AI Vision Model",
    [
        "google/gemini-2.5-flash",        # Proven Baseline (~$0.006/scan)
        "openai/gpt-5.6-luna",            # Most Popular / High Detail (~$0.003/scan)
        "z-ai/glm-5.3-flash",             # Lowest Cost (~$0.0008/scan)
        "minimax/minimax-m3"              # Lowest Latency / 516ms (~$0.002/scan)
    ],
    index=0,
    help="Gemini 2.5 Flash (Proven accuracy), GPT-5.6 Luna (#1 volume), GLM 5.3 Flash (Lowest cost), MiniMax M3 (Fastest)."
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

enable_web_enrichment = st.sidebar.checkbox(
    "🌐 Live Web Search Enrichment (15 Workers)",
    value=True,
    help="After scanning, runs 15 parallel Gemini 2.5 Flash (:online) workers to look up exact publisher sales figures, TV/screen adaptations, and romance/spice ratings for each unique book."
)

use_parallel = st.sidebar.checkbox(
    "⚡ Turbo Parallel Mode (Multi-Threaded)", 
    value=False,
    help="Splits the image into horizontal shelf bands and scans them simultaneously with parallel workers (~4x faster)."
)

if use_parallel:
    auto_detect_shelves = st.sidebar.checkbox(
        "🪵 Auto-Detect Shelf Planks (OpenCV)",
        value=True,
        help="Automatically finds physical horizontal wooden shelves using edge detection, avoiding cutting books in half."
    )
    if not auto_detect_shelves:
        num_parallel_shelves = st.sidebar.slider(
            "Estimated Shelves in Photo",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
            help="Number of parallel workers to launch. Matches the number of shelf rows."
        )
    else:
        num_parallel_shelves = None
else:
    auto_detect_shelves = False
    num_parallel_shelves = 1

if st.sidebar.button("🗑️ Reset / Clear All"):
    st.session_state.processed_images = {}
    st.session_state.master_books = []
    st.session_state.pending_uploads = {}
    # Bumping the nonce rebuilds the uploader widget, which is the only way to
    # drop files it is already holding.
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
    [
        "🌟 Author Fame & Lifetime Sales (Within Genre)",
        "Most Sales / Popularity (Book Volume)", 
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
        pil_img.save(buf, format="JPEG", quality=85, optimize=True)
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
            if x_inter / float(max(w1, w2) + 1e-5) > 0.75:
                v_gap = max(box1[0], box2[0]) - min(box1[2], box2[2])
                if v_gap <= 40:
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
                if (v_overlap > 0 or v_gap < 25) and (h_b < h_k * 0.55 or h_k < h_b * 0.55):
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
    """Detect horizontal wooden shelf planks using Sobel edge filter, CLAHE, and prominence peak detection."""
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

    # Peak prominence detection in pure numpy (scale-invariant)
    min_dist = int(det_h * 0.09)
    min_prom = float(np.max(smoothed)) * 0.08
    candidates = []
    for i in range(int(det_h * 0.10), int(det_h * 0.94)):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            # Left trough
            l = i - 1
            while l > 0 and smoothed[l] <= smoothed[i]:
                l -= 1
            l_min = np.min(smoothed[l:i])
            # Right trough
            r = i + 1
            while r < det_h - 1 and smoothed[r] <= smoothed[i]:
                r += 1
            r_min = np.min(smoothed[i + 1:r + 1])
            prom = smoothed[i] - max(l_min, r_min)
            if prom >= min_prom:
                candidates.append((i, prom))

    candidates.sort(key=lambda x: x[1], reverse=True)
    filtered = []
    for p, prom in candidates:
        if all(abs(p - f) >= min_dist for f in filtered):
            filtered.append(p)
    peaks = sorted(filtered)

    # If expected_shelves is specified and more peaks found, retain the most prominent
    if expected_shelves is not None and expected_shelves > 1 and len(peaks) > (expected_shelves - 1):
        prom_dict = dict(candidates)
        sorted_peaks = sorted(peaks, key=lambda p: prom_dict.get(p, 0), reverse=True)[:expected_shelves - 1]
        peaks = sorted(sorted_peaks)

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


def process_bookshelf(img_bytes, image_id, image_name, mode, model_id, key, status_cb=None, use_parallel=False, num_shelves=4, auto_detect_shelves=True):
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
        if use_parallel:
            crops_data = []
            slice_bounds = []
            if auto_detect_shelves:
                planks = detect_shelf_planks(img, expected_shelves=num_shelves if (num_shelves and num_shelves > 1) else None)
                if planks:
                    slice_bounds = compute_shelf_slices(H, W, planks)
                    status_cb(f"🪵 Auto-detected {len(slice_bounds)} physical shelves from horizontal planks! Launching parallel workers…")
            
            if not slice_bounds:
                n_s = num_shelves if (num_shelves and num_shelves > 1) else 4
                slice_bounds = compute_equal_slices(H, W, n_s)
                status_cb(f"🚀 Launching {len(slice_bounds)} parallel shelf workers simultaneously…")
                
            started_p = time.time()
            for s_idx, (y1, y2) in enumerate(slice_bounds, start=1):
                crops_data.append((s_idx, img[y1:y2, :], y1, y2 - y1))

            def _worker(args):
                s_idx, crop_img, y_off, ch = args
                res = call_vision_api(crop_img, model_id, key, status_cb=None)
                remapped = []
                for ab in res:
                    ymin, xmin, ymax, xmax = ab.get("box_2d", [0, 0, 0, 0])
                    # Filter partial sliver fragments peeking across the top or bottom border of the shelf crop
                    box_h_pct = (ymax - ymin) / 1000.0
                    if (ymin <= 40 and box_h_pct < 0.35) or (ymax >= 960 and box_h_pct < 0.35):
                        continue
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
                "title": str(ab.get("title") or f"Book {idx}"),
                "author": str(ab.get("author") or "Unknown"),
                "spine_text": str(ab.get("spine_text") or ""),
                "category": ab.get("category", "Standalone Novel"),
                "series": ab.get("series_info", "-"),
                "protagonist": ab.get("protagonist", "-"),
                "sensual_romance_flag": ab.get("sensual_flag", "✔️ None (Pure Thriller / Mystery)"),
                "tv_adaptation": ab.get("tv_adaptation", "❌ No"),
                "sales": ab.get("sales_popularity", "Standard"),
                "sales_score": 10.0 if "Bestseller" in ab.get("sales_popularity", "") else 1.0,
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
                    "sensual_romance_flag": "✔️ None (Pure Thriller / Mystery)",
                    "tv_adaptation": "❌ No",
                    "sales": "Standard",
                    "sales_score": 1.0,
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


def render_model_arena(api_key, scanner_mode):
    st.markdown("### ⚔️ AI Vision Model Shootout Arena")
    st.caption("Benchmark all 4 models on the exact same bookshelf photo under real-world Streamlit conditions (measuring total end-to-end wall-clock time including upload, API latency, token streaming, shelf plank detection, and visual box rendering).")

    queued_keys = list(st.session_state.pending_uploads.keys())
    queued = [st.session_state.pending_uploads[k] for k in queued_keys]

    arena_col1, arena_col2 = st.columns([1.4, 1.0])
    
    with arena_col1:
        st.markdown("#### 1. Select Bookshelf Image to Test")
        source_options = ["Example Bookstore Shelf (Pre-bundled)"]
        if queued:
            source_options.insert(0, f"Current Queued Photo ({queued[0][0]})")
        source_options.append("Upload a New Photo for Shootout")
        
        arena_img_source = st.radio("Image Source", source_options, index=0)
        
        arena_img_bytes = None
        arena_img_name = "Benchmark_Shelf.jpg"
        
        if arena_img_source.startswith("Current Queued Photo") and queued:
            arena_img_name, arena_img_bytes = queued[0]
        elif arena_img_source == "Upload a New Photo for Shootout":
            up_arena = st.file_uploader("Upload bookshelf photo for arena", type=UPLOAD_TYPES, key="arena_uploader")
            if up_arena is not None:
                arena_img_bytes = downscale_ingest_bytes(up_arena.getvalue())
                arena_img_name = up_arena.name
        else:
            if os.path.exists(SAMPLE_IMAGE):
                with open(SAMPLE_IMAGE, "rb") as f:
                    arena_img_bytes = f.read()
                arena_img_name = "Sample_Shelf.jpg"

        if arena_img_bytes:
            st.image(arena_img_bytes, caption=f"Selected: {arena_img_name} ({len(arena_img_bytes)//1024} KB)", width=320)

    with arena_col2:
        st.markdown("#### 2. Select Models to Benchmark")
        test_gemini = st.checkbox("Google Gemini 2.5 Flash", value=True)
        test_gpt = st.checkbox("OpenAI GPT-5.6 Luna", value=True)
        test_glm = st.checkbox("Z-AI GLM 5.3 Flash", value=True)
        test_minimax = st.checkbox("MiniMax M3", value=True)
        
        st.markdown("#### 3. Execution Settings")
        arena_parallel = st.checkbox("⚡ Use Turbo Parallel Mode (Physical Shelf Planks)", value=True,
                                     help="Tests models using concurrent shelf workers sliced along physical wooden planks.")

    selected_arena_models = []
    if test_gemini: selected_arena_models.append(("google/gemini-2.5-flash", "Gemini 2.5 Flash"))
    if test_gpt: selected_arena_models.append(("openai/gpt-5.6-luna", "GPT-5.6 Luna"))
    if test_glm: selected_arena_models.append(("z-ai/glm-5.3-flash", "GLM 5.3 Flash (Z-AI)"))
    if test_minimax: selected_arena_models.append(("minimax/minimax-m3", "MiniMax M3"))

    st.markdown("---")
    if not api_key:
        st.warning("⚠️ Please connect or enter your OpenRouter API key in the sidebar before running the shootout.")
    elif arena_img_bytes is None:
        st.info("ℹ️ Please select or upload an image to start.")
    elif not selected_arena_models:
        st.warning("⚠️ Please select at least one model to benchmark.")
    else:
        if st.button("🚀 Launch Model Shootout", type="primary", width="stretch"):
            st.session_state.arena_results = {}
            progress_bar = st.progress(0.0)
            arena_status = st.empty()
            
            for idx, (m_id, m_label) in enumerate(selected_arena_models, 1):
                arena_status.info(f"⏳ **Testing {idx}/{len(selected_arena_models)}: {m_label}** — Running full end-to-end pipeline…")
                t0 = time.time()
                try:
                    pil_res, b_out, err = process_bookshelf(
                        arena_img_bytes, idx, f"{m_label} Benchmark", scanner_mode,
                        m_id, api_key,
                        status_cb=lambda msg, lbl=m_label: arena_status.info(f"⏳ **{lbl}**: {msg}"),
                        use_parallel=arena_parallel,
                        num_shelves=None,
                        auto_detect_shelves=True
                    )
                    t_total = time.time() - t0
                    if err:
                        st.session_state.arena_results[m_label] = {
                            "model_id": m_id,
                            "time": round(t_total, 2),
                            "books": 0,
                            "identified": 0,
                            "unidentified": 0,
                            "error": err,
                            "image": None,
                            "book_list": []
                        }
                    else:
                        unidentified = sum(1 for b in b_out if "Unidentified" in b.get("title", ""))
                        st.session_state.arena_results[m_label] = {
                            "model_id": m_id,
                            "time": round(t_total, 2),
                            "books": len(b_out),
                            "identified": len(b_out) - unidentified,
                            "unidentified": unidentified,
                            "error": None,
                            "image": pil_res,
                            "book_list": b_out
                        }
                except Exception as ex:
                    t_total = time.time() - t0
                    st.session_state.arena_results[m_label] = {
                        "model_id": m_id,
                        "time": round(t_total, 2),
                        "books": 0,
                        "identified": 0,
                        "unidentified": 0,
                        "error": str(ex),
                        "image": None,
                        "book_list": []
                    }
                progress_bar.progress(idx / len(selected_arena_models))
                
            arena_status.success("🎉 Shootout complete! See leaderboard and visual comparisons below.")
            progress_bar.empty()

    # Display Leaderboard and Side-by-Side Images
    if st.session_state.get("arena_results"):
        st.markdown("---")
        st.markdown("### 🏆 Arena Leaderboard & Timing Results")
        
        leaderboard = []
        sorted_results = sorted(st.session_state.arena_results.items(), key=lambda x: (x[1]["time"] if not x[1]["error"] else 9999))
        
        valid_times = [r["time"] for _, r in sorted_results if not r["error"]]
        fastest_time = min(valid_times) if valid_times else None
        valid_books = [r["books"] for _, r in sorted_results if not r["error"]]
        max_books = max(valid_books) if valid_books else None
        
        for rank, (m_label, r) in enumerate(sorted_results, 1):
            if r["error"]:
                speed_str = "Error"
            else:
                speed_str = f"{r['time']}s"
                if r['time'] == fastest_time:
                    speed_str += " ⚡ (Fastest)"
                    
            book_str = str(r["books"])
            if r["books"] == max_books and r["books"] > 0:
                book_str += " 🏆 (Most)"
                
            leaderboard.append({
                "Rank": f"#{rank}",
                "Model": m_label,
                "Total Wall Time": speed_str,
                "Books Detected": book_str,
                "Legible Spines": r["identified"],
                "Unidentified": r["unidentified"],
                "Status": "✅ OK" if not r["error"] else f"❌ {r['error']}"
            })
        st.dataframe(leaderboard, width="stretch")
        
        st.markdown("### 🔍 Side-by-Side Visual Quality Inspector")
        st.caption("Inspect and zoom into each model's output image to check bounding box accuracy, spine coverage, and text legibility.")
        
        model_tabs = st.tabs([m_label for m_label, _ in sorted_results])
        for tab, (m_label, r) in zip(model_tabs, sorted_results):
            with tab:
                if r["image"] is not None:
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Total End-to-End Time", f"{r['time']}s")
                    col_m2.metric("Books Detected", f"{r['books']}")
                    col_m3.metric("Legible Titles", f"{r['identified']}")
                    render_zoomable_image(r["image"], height=620)
                    
                    with st.expander(f"📋 View Detected Books List ({r['books']} books)"):
                        tbl = [{
                            "#": b.get("id"),
                            "Shelf": b.get("shelf"),
                            "Spine Text": b.get("spine_text") or "-",
                            "Title": b.get("title"),
                            "Author": b.get("author")
                        } for b in r.get("book_list", [])]
                        st.dataframe(tbl, width="stretch")
                else:
                    st.error(f"Execution failed for {m_label}: {r.get('error')}")




def execute_model_search(model_id, title, author, api_key):
    """Execute live web search for a book using specified OpenRouter search model."""
    is_sonar = "sonar" in model_id.lower()
    prompt = f"""You are an objective book industry cataloging agent. Perform a live web search for the published book '{title}' by author '{author}'.
Search and extract:
1. "book_sales": Verified volume for THIS specific book across ALL formats: print copies sold, Kindle/ebook downloads, and audiobook listens (e.g., 'Over 6.5 million copies sold', '1.8M across print, digital, and audio'). If no official book-specific count is publicly reported on the web, output strictly 'Not publicly reported'. NEVER guess or invent numbers.
2. "author_fame": The author's total lifetime career sales / reader reach across all their books and formats (e.g., 'Over 60 million books worldwide', '70M+ In Death series copies', 'Over 400M books sold', or 'Debut / Emerging Author'). If unknown, write 'Not publicly reported'.
3. "author_fame_score": Numeric total author copies sold (e.g. 60000000 for 60M, 70000000 for 70M, 0 if unknown) for ranking within genre.
4. "tv_adaptation": Has this book or series been adapted or optioned for TV or film? State: 'Yes (Network/Title)', 'Optioned / In Dev (Studio)', or 'No'.
5. "sensual_rating": Content rating: 'Explicit Romance / Sensual', 'Moderate Romance', or 'Clean / None (Pure Mystery/Thriller)'.

Return strictly a valid JSON object:
{{
  "book_sales": "...",
  "author_fame": "...",
  "author_fame_score": 0,
  "tv_adaptation": "...",
  "sensual_rating": "...",
  "evidence": "1-sentence summary of factual search source"
}}"""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    if not is_sonar:
        payload["plugins"] = [{"id": "web"}]
        payload["response_format"] = {"type": "json_object"}

    t0 = time.time()
    try:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=_api_headers(api_key),
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = _extract_json(content) or {"raw": content}
            parsed["latency"] = round(time.time() - t0, 2)
            return parsed
    except Exception as e:
        return {"error": str(e), "latency": round(time.time() - t0, 2)}


def enrich_catalog_in_parallel(books, api_key, status_cb=None, max_workers=15):
    """Enrich detected books using google/gemini-2.5-flash:online across unique titles with 15 parallel workers."""
    if not api_key or not books:
        return books

    status_cb = status_cb or (lambda _msg: None)

    unique_map = {}
    for b in books:
        t = (b.get("title") or "").strip()
        a = (b.get("author") or "").strip()
        if not t or "unidentified" in t.lower() or t.lower().startswith("book "):
            continue
        canon_key = get_canonical_key(t, a)
        if canon_key not in unique_map:
            unique_map[canon_key] = {"title": t, "author": a}

    if not unique_map:
        return books

    total_unique = len(unique_map)
    status_cb(f"🚀 Launching 15 parallel search workers for {total_unique} unique titles…")

    enrichment_cache = {}
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(execute_model_search, "google/gemini-2.5-flash:online", info["title"], info["author"], api_key): c_key
            for c_key, info in unique_map.items()
        }
        for future in concurrent.futures.as_completed(future_to_key):
            c_key = future_to_key[future]
            try:
                res = future.result()
                enrichment_cache[c_key] = res
            except Exception as ex:
                enrichment_cache[c_key] = {"error": str(ex)}
            done_count += 1
            if done_count % 3 == 0 or done_count == total_unique:
                status_cb(f"🌐 Enriched {done_count}/{total_unique} titles with live web search…")

    for b in books:
        t = (b.get("title") or "").strip()
        a = (b.get("author") or "").strip()
        canon_key = get_canonical_key(t, a)
        if canon_key in enrichment_cache:
            e = enrichment_cache[canon_key]
            if not e.get("error"):
                # Book-specific sales across all formats
                b_sales = e.get("book_sales") or e.get("exact_sales") or "Not publicly reported"
                b["sales"] = b_sales
                b_str = str(b_sales).lower()
                if "million" in b_str:
                    num_match = re.search(r'([\d\.]+)\s*million', b_str)
                    b["sales_score"] = float(num_match.group(1)) * 1_000_000 if num_match else 1_000_000.0
                elif bool(re.search(r'\d', b_str)) and "not publicly" not in b_str:
                    digits = re.sub(r'[^\d]', '', b_str)
                    b["sales_score"] = float(digits) if digits else 10.0
                else:
                    b["sales_score"] = 1.0

                # Author Fame (Career lifetime sales)
                a_fame = e.get("author_fame") or "Not publicly reported"
                b["author_fame"] = a_fame
                fame_val = e.get("author_fame_score", 0)
                if not isinstance(fame_val, (int, float)) or fame_val <= 0:
                    a_str = str(a_fame).lower()
                    if "billion" in a_str:
                        m = re.search(r'([\d\.]+)\s*billion', a_str)
                        fame_val = float(m.group(1)) * 1_000_000_000 if m else 1_000_000_000.0
                    elif "million" in a_str:
                        m = re.search(r'([\d\.]+)\s*million', a_str)
                        fame_val = float(m.group(1)) * 1_000_000 if m else 1_000_000.0
                    elif bool(re.search(r'\d', a_str)) and "not publicly" not in a_str:
                        digs = re.sub(r'[^\d]', '', a_str)
                        fame_val = float(digs) if digs else 0.0
                    else:
                        fame_val = 0.0
                b["author_fame_score"] = float(fame_val)

                if "tv_adaptation" in e and e["tv_adaptation"]:
                    b["tv_adaptation"] = e["tv_adaptation"]
                if "sensual_rating" in e and e["sensual_rating"]:
                    b["sensual_romance_flag"] = e["sensual_rating"]
                if "evidence" in e and e["evidence"]:
                    b["search_evidence"] = e["evidence"]

    return books


BENCHMARK_25_BOOKS = [
    {"id": 1, "title": "The Silent Patient", "author": "Alex Michaelides", "type": "Psychological Thriller"},
    {"id": 2, "title": "The Housemaid", "author": "Freida McFadden", "type": "Domestic Thriller"},
    {"id": 3, "title": "The Chain", "author": "Adrian McKinty", "type": "Kidnapping Thriller"},
    {"id": 4, "title": "Still Life", "author": "Louise Penny", "type": "Police Procedural"},
    {"id": 5, "title": "The Whisper Man", "author": "Alex North", "type": "Serial Killer Thriller"},
    {"id": 6, "title": "Behind Closed Doors", "author": "B.A. Paris", "type": "Psychological Suspense"},
    {"id": 7, "title": "The Collector", "author": "Daniel Silva", "type": "Espionage Thriller"},
    {"id": 8, "title": "The Devil's Star", "author": "Jo Nesbo", "type": "Nordic Noir"},
    {"id": 9, "title": "Naked in Death", "author": "J.D. Robb", "type": "Romantic Suspense (Explicit)"},
    {"id": 10, "title": "The Rivals", "author": "Jane Pek", "type": "Amateur Sleuth (Moderate Romance)"},
    {"id": 11, "title": "Gone Girl", "author": "Gillian Flynn", "type": "Mega Bestseller / Psychological"},
    {"id": 12, "title": "Where the Crawdads Sing", "author": "Delia Owens", "type": "Mega Bestseller / Mystery"},
    {"id": 13, "title": "The Thursday Murder Club", "author": "Richard Osman", "type": "Cozy Mystery"},
    {"id": 14, "title": "Verity", "author": "Colleen Hoover", "type": "Romantic Thriller (Explicit)"},
    {"id": 15, "title": "The Girl on the Train", "author": "Paula Hawkins", "type": "Mega Bestseller / Thriller"},
    {"id": 16, "title": "Dark Matter", "author": "Blake Crouch", "type": "Sci-Fi Thriller"},
    {"id": 17, "title": "The Guest List", "author": "Lucy Foley", "type": "Whodunit Murder Mystery"},
    {"id": 18, "title": "Wrong Place Wrong Time", "author": "Gillian McAllister", "type": "Time-Loop Thriller"},
    {"id": 19, "title": "None of This Is True", "author": "Lisa Jewell", "type": "Podcast Crime Thriller"},
    {"id": 20, "title": "I Am Pilgrim", "author": "Terry Hayes", "type": "Espionage Blockbuster"},
    {"id": 21, "title": "First Lie Wins", "author": "Ashley Elston", "type": "Con-Artist Thriller"},
    {"id": 22, "title": "A Good Girl's Guide to Murder", "author": "Holly Jackson", "type": "YA Mystery / Procedural"},
    {"id": 23, "title": "The Maid", "author": "Nita Prose", "type": "Hotel Mystery"},
    {"id": 24, "title": "Local Woman Missing", "author": "Mary Kubica", "type": "Missing Persons Mystery"},
    {"id": 25, "title": "Surprise Me", "author": "Sophie Kinsella", "type": "Romantic Comedy (Moderate Romance)"},
]


def fetch_supercharged_free_context(title, author):
    """Fetch rich, targeted free snippets from Wikipedia REST API and DuckDuckGo fanout."""
    snippets = []

    # 1. Wikipedia Search REST API (zero cost, exact infobox & summary data)
    for q in [f"{title} {author}", f"{author} author"]:
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(q)}&format=json&utf8="
            req = urllib.request.Request(url, headers={"User-Agent": "BookshelfScanner/2.0 (contact@bookshelfai.com)"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                d = json.loads(resp.read().decode("utf-8"))
                for item in d.get("query", {}).get("search", [])[:2]:
                    clean = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                    snippets.append(f"[Wikipedia: {item.get('title')}] {clean}")
        except Exception:
            pass

    # 2. DuckDuckGo 3 Targeted Queries (sales volume, author fame, screen adaptation)
    try:
        from duckduckgo_search import DDGS
        ddg = DDGS()
        queries = [
            f'"{title}" "{author}" copies sold OR million',
            f'"{author}" books sold worldwide OR career sales OR million copies',
            f'"{title}" "{author}" film adaptation OR TV series OR movie'
        ]
        for q in queries:
            try:
                for r in ddg.text(q, max_results=2):
                    t = (r.get("title") or "").strip()
                    b = (r.get("body") or "").strip()
                    if t or b:
                        snippets.append(f"[{t}] {b}")
            except Exception:
                pass
    except Exception:
        pass

    return "\n".join(snippets)


def execute_option_b_search(title, author, api_key):
    """Option B: Fetch free web & Wikipedia context, then use standard Gemini 2.5 Flash ($0 search fee)."""
    t0 = time.time()
    free_ctx = fetch_supercharged_free_context(title, author)

    prompt = f"""You are an objective book industry cataloging agent.
Below is search context retrieved from Wikipedia and public web sources for the published book '{title}' by author '{author}':
---
{free_ctx if free_ctx.strip() else "No public search snippets retrieved."}
---

Based on the provided search snippets AND your factual knowledge of published literature, extract:
1. "book_sales": Verified volume for THIS specific book across ALL formats: print copies sold, Kindle/ebook downloads, and audiobook listens (e.g., 'Over 6.5 million copies sold', '1.8M across print, digital, and audio'). If no official book-specific count is publicly reported on the web, output strictly 'Not publicly reported'. NEVER guess or invent numbers. NEVER confuse author career total with this book's sales.
2. "author_fame": The author's total lifetime career sales / reader reach across all their books and formats (e.g., 'Over 60 million books worldwide', '70M+ In Death series copies', 'Over 400M books sold', or 'Debut / Emerging Author'). If unknown, write 'Not publicly reported'.
3. "author_fame_score": Numeric total author copies sold (e.g. 60000000 for 60M, 70000000 for 70M, 0 if unknown) for ranking within genre.
4. "tv_adaptation": Has this book or series been adapted or optioned for TV or film? State: 'Yes (Network/Title)', 'Optioned / In Dev (Studio)', or 'No'.
5. "sensual_rating": Content rating: 'Explicit Romance / Sensual', 'Moderate Romance', or 'Clean / None (Pure Mystery/Thriller)'.
6. "evidence": 1-sentence summary of factual search source.

Return strictly a valid JSON object:
{{
  "book_sales": "...",
  "author_fame": "...",
  "author_fame_score": 0,
  "tv_adaptation": "...",
  "sensual_rating": "...",
  "evidence": "..."
}}"""

    payload = {
        "model": "google/gemini-2.5-flash",  # Standard model: ZERO search query fee ($0.00)
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    try:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=_api_headers(api_key),
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = _extract_json(content) or {"raw": content}
            parsed["latency"] = round(time.time() - t0, 2)
            parsed["cost_est"] = 0.00015
            return parsed
    except Exception as e:
        return {"error": str(e), "latency": round(time.time() - t0, 2), "cost_est": 0.0}


def render_enrichment_arena_25(api_key):
    st.markdown("### 🔍 25-Book Search Arena: Option A vs Supercharged Option B")
    st.caption("Benchmark Option A (OpenRouter Live Web Search with $0.007 query fee) vs Supercharged Option B (Free Wikipedia API + DDG 3-Query Fanout + Standard Gemini at $0 search fee).")

    if not api_key:
        st.warning("⚠️ Please connect your OpenRouter API key in the left sidebar.")
        return

    col1, col2 = st.columns([1.5, 1.0])
    with col1:
        st.markdown("#### 1. Strategies to Benchmark")
        run_a = st.checkbox("Option A: OpenRouter Live Web Search (Gemini 2.5 Flash :online, ~$0.008/book)", value=True, key="chk_opt_a")
        run_b = st.checkbox("Supercharged Option B: Free Wikipedia + DDG Fanout + Gemini ($0 search fee, ~$0.00015/book)", value=True, key="chk_opt_b")
    with col2:
        st.markdown("#### 2. Test Configuration")
        book_count = st.radio("Number of Books to Search", [10, 20, 25], index=2, horizontal=True)
        concurrency = st.slider("Parallel Worker Threads", min_value=1, max_value=15, value=12, key="arena_threads")

    test_books = BENCHMARK_25_BOOKS[:book_count]

    with st.expander(f"📚 View Active Test Dataset ({len(test_books)} Books)"):
        st.dataframe(test_books, width="stretch", hide_index=True)

    active_modes = []
    if run_a:
        active_modes.append(("Option A (OpenRouter Search)", "a"))
    if run_b:
        active_modes.append(("Option B (Free Fanout + Gemini)", "b"))

    if not active_modes:
        st.warning("Select at least one strategy to run.")
        return

    if st.button("🚀 Run 25-Book Shootout (Option A vs Option B)", type="primary", key="btn_run_ab_shootout"):
        all_results = {}
        timings = {}
        total_steps = len(active_modes) * len(test_books)
        completed_steps = 0
        prog_bar = st.progress(0)
        status_box = st.empty()

        for label, mode_key in active_modes:
            status_box.info(f"🌐 Running **{label}** across {len(test_books)} books with {concurrency} parallel workers...")
            t_start = time.time()
            m_res = {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                if mode_key == "a":
                    futures = {
                        executor.submit(execute_model_search, "google/gemini-2.5-flash:online", b["title"], b["author"], api_key): b["id"]
                        for b in test_books
                    }
                else:
                    futures = {
                        executor.submit(execute_option_b_search, b["title"], b["author"], api_key): b["id"]
                        for b in test_books
                    }

                for f in concurrent.futures.as_completed(futures):
                    b_id = futures[f]
                    try:
                        m_res[b_id] = f.result()
                    except Exception as ex:
                        m_res[b_id] = {"error": str(ex), "latency": 0.0}
                    completed_steps += 1
                    prog_bar.progress(int((completed_steps / float(total_steps)) * 100))

            timings[label] = time.time() - t_start
            all_results[label] = m_res

        prog_bar.progress(100)
        status_box.success("✅ Shootout Complete!")

        # Performance & Cost Leaderboard
        st.markdown("---")
        st.markdown("#### ⚡ Speed & Cost Leaderboard")
        m_cols = st.columns(len(active_modes))
        for idx, (label, mode_key) in enumerate(active_modes):
            t_tot = timings.get(label, 0.0)
            avg_lat = t_tot / float(len(test_books)) if test_books else 0.0
            cost_test = len(test_books) * (0.0082 if mode_key == "a" else 0.00015)
            cost_168 = 168 * (0.0082 if mode_key == "a" else 0.00015)
            with m_cols[idx]:
                st.metric(label, f"{t_tot:.2f}s total", f"{avg_lat:.2f}s / book")
                st.caption(f"Cost for this test: **${cost_test:.3f}** • Projected 168-book shelf: **${cost_168:.2f}**")

        # Side-by-Side Content Comparison Table
        st.markdown("---")
        st.markdown("#### 📊 Side-by-Side Content Comparison")
        table_rows = []
        for b in test_books:
            bid = b["id"]
            row = {
                "#": bid,
                "Book": f"{b['title']} — {b['author']}",
                "Genre": b.get("type", "-"),
            }
            for label, _ in active_modes:
                res = all_results.get(label, {}).get(bid, {})
                sales = res.get("book_sales") or res.get("exact_sales") or "-"
                fame = res.get("author_fame") or "-"
                tv = res.get("tv_adaptation") or "-"
                sensual = res.get("sensual_rating") or "-"
                lat = res.get("latency", 0.0)
                prefix = "Opt A" if "Option A" in label else "Opt B"
                row[f"{prefix}: Sales"] = sales
                row[f"{prefix}: Author Fame"] = fame
                row[f"{prefix}: TV"] = tv
                row[f"{prefix}: Sensual"] = sensual
                row[f"{prefix}: Latency"] = f"{lat:.1f}s"
            table_rows.append(row)

        st.dataframe(table_rows, width="stretch")

        # Copy-Paste Section
        st.markdown("---")
        st.markdown("#### 📋 Copy-Paste Output Block (For Verification)")

        copy_text_lines = []
        copy_text_lines.append(f"# 25-Book Shootout Results: Option A vs Supercharged Option B\n")
        for label, _ in active_modes:
            copy_text_lines.append(f"## Strategy: {label} (Time: {timings.get(label, 0):.2f}s across {len(test_books)} books)\n")
        copy_text_lines.append("=" * 60 + "\n")

        for b in test_books:
            bid = b["id"]
            copy_text_lines.append(f"### {bid}. {b['title']} by {b['author']} ({b['type']})")
            for label, _ in active_modes:
                res = all_results.get(label, {}).get(bid, {})
                copy_text_lines.append(f"**[{label}]**:")
                copy_text_lines.append(json.dumps(res, indent=2))
            copy_text_lines.append("\n" + "-" * 40 + "\n")

        full_copy_text = "\n".join(copy_text_lines)
        st.text_area("Select All & Copy (Ctrl+A, Ctrl+C):", full_copy_text, height=350)

        with st.expander("🔍 View Formatted JSON by Book"):
            for b in test_books:
                bid = b["id"]
                st.markdown(f"##### {bid}. {b['title']} by {b['author']}")
                cols = st.columns(len(active_modes))
                for idx, (label, _) in enumerate(active_modes):
                    with cols[idx]:
                        st.caption(f"**{label}**")
                        st.json(all_results.get(label, {}).get(bid, {}))
                st.markdown("---")


tab_scanner, tab_arena, tab_shootout = st.tabs([
    "📚 Shelf Scanner & Cataloger",
    "⚔️ 4-Model Shootout Arena",
    "🔍 25-Book Search Arena (Option A vs Option B)"
])

with tab_arena:
    render_model_arena(api_key, scanner_mode)

with tab_shootout:
    render_enrichment_arena_25(api_key)




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
                    st.session_state.uploader_nonce += 1
                    st.session_state.last_client_batch_id = ""
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
                pil_img, books, err = process_bookshelf(
                    img_bytes, idx, f"Image {idx}", scanner_mode,
                    selected_model, api_key, status_cb,
                    use_parallel=use_parallel, num_shelves=num_parallel_shelves,
                    auto_detect_shelves=auto_detect_shelves,
                )
                if err:
                    failures.append(f"**{name}** — {err}")
                elif pil_img is not None:
                    st.session_state.processed_images[f"Image {idx} ({name})"] = pil_img
                    st.session_state.master_books.extend(books)
                progress.progress(idx / len(queued))

            if enable_web_enrichment and st.session_state.master_books and api_key:
                status.info("🌐 Launching 15 parallel Gemini 2.5 Flash (:online) search workers to look up exact sales, TV adaptations, and romance ratings…")
                st.session_state.master_books = enrich_catalog_in_parallel(
                    st.session_state.master_books,
                    api_key,
                    status_cb=lambda msg: status.info(f"🌐 **Live Web Search Agent**\n\n{msg}"),
                    max_workers=15
                )

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
                use_parallel=use_parallel, num_shelves=num_parallel_shelves,
                auto_detect_shelves=auto_detect_shelves,
            )
            if err:
                status.empty()
                st.error(f"❌ Example shelf — {err}")
            else:
                st.session_state.processed_images = {"Image 1 (Example Bookstore Shelf)": pil_img}
                st.session_state.master_books = books
                if enable_web_enrichment and api_key:
                    status.info("🌐 Launching 15 parallel Gemini 2.5 Flash (:online) search workers to look up exact sales, TV adaptations, and romance ratings…")
                    st.session_state.master_books = enrich_catalog_in_parallel(
                        st.session_state.master_books,
                        api_key,
                        status_cb=lambda msg: status.info(f"🌐 **Live Web Search Agent**\n\n{msg}"),
                        max_workers=15
                    )
                status.empty()
                st.success(f"🎉 Example shelf loaded: {len(st.session_state.master_books)} books identified & enriched!")
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
            flag = str(b.get("sensual_romance_flag", "")).lower()
            if sensual_filter == "✔️ Clean Only (No Explicit Romance)" and ("explicit" in flag or "❌" in flag):
                continue
            if sensual_filter == "❌ Explicit Romance / Sensual Only" and ("explicit" not in flag and "❌" not in flag):
                continue
            
            tv = str(b.get("tv_adaptation", "")).lower()
            is_adapted = tv.startswith("yes") or "in dev" in tv or "optioned" in tv or "📺" in tv
            if tv_filter == "📺 TV / Screen Adapted Only" and not is_adapted:
                continue
            if tv_filter == "❌ Non-Adapted Only" and is_adapted:
                continue
            
            if cat_filter != "All Categories" and b.get("category", "") != cat_filter:
                continue
            
            filtered_books.append(b)

        # Sort
        if sort_by == "🌟 Author Fame & Lifetime Sales (Within Genre)":
            filtered_books.sort(
                key=lambda x: (
                    str(x.get("category") or "Standalone Novel"),
                    -float(x.get("author_fame_score", 0.0)),
                    -float(x.get("sales_score", 0.0))
                )
            )
        elif sort_by == "Most Sales / Popularity (Book Volume)":
            filtered_books.sort(key=lambda x: x.get("sales_score", 0.0), reverse=True)
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
        m3.metric("TV / Screen Adapted", f"{sum(1 for b in filtered_books if str(b.get('tv_adaptation','')).lower().startswith('yes') or 'in dev' in str(b.get('tv_adaptation','')).lower() or 'optioned' in str(b.get('tv_adaptation','')).lower() or '📺' in str(b.get('tv_adaptation','')))}")
        m4.metric("Non-Adapted", f"{sum(1 for b in filtered_books if not (str(b.get('tv_adaptation','')).lower().startswith('yes') or 'in dev' in str(b.get('tv_adaptation','')).lower() or 'optioned' in str(b.get('tv_adaptation','')).lower() or '📺' in str(b.get('tv_adaptation',''))))}")

        st.markdown("---")

        img_col, table_col = st.columns([1, 1.2])

        with img_col:
            st.subheader("📷 Shelf Highlights")
            if st.session_state.processed_images:
                img_choice = st.selectbox("Select Image to Inspect", list(st.session_state.processed_images.keys()))
                selected_img = st.session_state.processed_images[img_choice]
                tab_zoom, tab_static = st.tabs(["🔍 Interactive Zoom & Pan", "🖼️ Overview"])
                with tab_zoom:
                    render_zoomable_image(selected_img, height=620)
                with tab_static:
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
                    "Author Fame": b.get("author_fame", "Not publicly reported"),
                    "Book Sales / Listens": b.get("sales", "Not publicly reported"),
                    "Genre / Category": b.get("category", "Standalone Novel"),
                    "Series / Protagonist": f"{b.get('series')} ({b.get('protagonist')})" if b.get("protagonist") != "-" else b.get("series"),
                    "TV Adaptation": b.get("tv_adaptation", "No"),
                    "Romance Flag": b.get("sensual_romance_flag", "Clean / None"),
                    "Search Evidence": b.get("search_evidence", "-"),
                    "Model": b.get("source", "API")
                })
            st.dataframe(table_rows, width="stretch", height=620)


st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Bookshelf AI • High Speed Vision")
