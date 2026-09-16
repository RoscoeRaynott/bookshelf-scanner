import streamlit as st
import cv2
import numpy as np
import json
import os
import re
import math
import base64
import urllib.request
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from rapidocr_onnxruntime import RapidOCR

st.set_page_config(
    page_title="Multi-Shelf Book Scanner & Cataloger",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Multi-Bookshelf Scanner & Master Cataloger")
st.caption("AI-Powered Book Scanner: Combines Gemini 2.5 Flash Vision with Offline OCR to detect, segment, and catalog every book on multi-column shelves.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGE = os.path.join(BASE_DIR, "data", "sample_shelf.jpg")
ANNOTATED_IMAGE = os.path.join(BASE_DIR, "data", "annotated_bookshelf_rotated.jpg")

@st.cache_resource
def get_ocr_engine():
    return RapidOCR(text_score=0.22)

ocr_engine = get_ocr_engine()

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

# Sidebar Settings
st.sidebar.header("🔑 API & Scanner Engine")

if detected_key:
    st.sidebar.success("✅ OpenRouter Key Connected (from Secrets)")
    api_key = detected_key
else:
    api_key = st.sidebar.text_input("Enter OpenRouter API Key", type="password", help="Enter key to use Gemini 2.5 Flash")

scanner_engine = st.sidebar.selectbox(
    "Select Detection Engine",
    [
        "⚡ Hybrid: Gemini 2.5 Flash + Offline OCR Bonus (Recommended)",
        "🌐 Gemini 2.5 Flash Only",
        "💻 Offline OCR Only"
    ]
)

deduplicate_catalog = st.sidebar.checkbox(
    "🔄 Deduplicate Overlaps & Repeat Sightings", 
    value=True,
    help="Merges overlapping books between adjacent shelves or multiple bookstores into a single entry with multi-location tags."
)

if st.sidebar.button("🗑️ Reset / Clear All"):
    st.session_state.processed_images = {}
    st.session_state.master_books = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Content & Story Filters")

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

# OpenRouter Vision API Caller
def call_gemini_vision(img_bgr, key):
    H, W, _ = img_bgr.shape
    # Encode to JPEG bytes
    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return []
    b64_img = base64.b64encode(buffer).decode("utf-8")
    
    prompt = """You are an expert bookstore cataloger.
Analyze this bookstore bookshelf image thoroughly. Locate every book visible across all shelves and bookcase sections.
For each book, identify its exact bounding box and metadata.
Return a valid JSON object with the exact format:
{
  "books": [
    {
      "box_2d": [ymin, xmin, ymax, xmax],
      "shelf_row": 1,
      "title": "Clean Canonical Title",
      "author": "Author Name",
      "category": "Strict Sequential Series" | "Recurring Protagonist" | "Standalone Novel",
      "series_info": "Series Name #Number or -",
      "protagonist": "Lead Character or -",
      "sensual_flag": "❌ Explicit Romance / Sensual" | "⚠️ Sensual Infidelity Elements" | "✔️ None (Pure Thriller / Mystery)",
      "tv_adaptation": "📺 Yes (Show Title / Network)" | "🎬 Optioned / In Prod." | "❌ No",
      "sales_popularity": "Estimated bestseller rank or Standard"
    }
  ]
}
Note on coordinates: "box_2d" must be normalized integers from 0 to 1000 representing [ymin, xmin, ymax, xmax].
Detect all books, including narrow vertical spines, leaning books, and face-out covers."""

    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}"
                        }
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bookshelf-scanner.streamlit.app",
            "X-Title": "Bookshelf Scanner"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content = resp_data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("books", [])
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return []

# Process Image Function
def process_bookshelf(img_bytes, image_id, image_name, mode, key):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        try:
            import io
            pil_t = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_t), cv2.COLOR_RGB2BGR)
        except Exception:
            return None, []
            
    H, W, _ = img.shape
    max_dim = max(H, W)
    if max_dim > 1600:
        scale = 1600.0 / max_dim
        img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    H, W, _ = img.shape
    
    books_out = []
    
    # 1. API Detection
    use_api = "Gemini" in mode and bool(key)
    api_books = []
    if use_api:
        api_books = call_gemini_vision(img, key)
        for idx, ab in enumerate(api_books, start=1):
            ymin, xmin, ymax, xmax = ab.get("box_2d", [0, 0, 0, 0])
            # Scale normalized 0-1000 to pixels
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
                "source": "Gemini 2.5 Flash"
            })
            
    # 2. Offline OCR Bonus Pass
    if "Offline" in mode or "Hybrid" in mode or not use_api:
        raw_results, _ = ocr_engine(img)
        if not use_api and raw_results:
            # Fallback pure offline detection
            for idx, item in enumerate(raw_results[:60], start=1):
                poly = np.array(item[0], dtype=np.int32)
                xmin, ymin = int(np.min(poly[:,0])), int(np.min(poly[:,1]))
                xmax, ymax = int(np.max(poly[:,0])), int(np.max(poly[:,1]))
                books_out.append({
                    "id": idx,
                    "image_id": image_id,
                    "image_name": image_name,
                    "shelf": 1,
                    "title": item[1][:30],
                    "author": "Offline Detected",
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
        
        # Badge
        badge_y = max(14, ymin)
        cv2.circle(annotated, (xmin + 8, badge_y), 8, (10, 10, 10), -1)
        cv2.putText(annotated, str(b["id"]), (xmin + 4 if b["id"] < 10 else xmin + 1, badge_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
                    
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
    pil_res = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    return pil_res, books_out

# Helper for deduplication key
def get_canonical_key(title, author):
    clean_t = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
    clean_a = re.sub(r'[^a-zA-Z0-9]', '', author.lower())
    return f"{clean_t}_{clean_a}"

# Session State
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}
if "master_books" not in st.session_state:
    st.session_state.master_books = []

# Main Upload Area
st.markdown("### 📸 Step 1: Select Photos")
col_up1, col_up2 = st.columns([3, 1])
with col_up1:
    uploaded_files = st.file_uploader(
        "Select one or more bookshelf photos from gallery or camera",
        accept_multiple_files=True
    )
with col_up2:
    st.write("")
    st.write("")
    use_demo = st.button("🧪 Test with Example Shelf", width="stretch")

if uploaded_files:
    st.write(f"📁 **{len(uploaded_files)} photo(s) selected.** Ready to scan.")
    if st.button("🚀 Run Scanner & Identify Books", type="primary", width="stretch"):
        st.session_state.processed_images = {}
        st.session_state.master_books = []
        
        for idx, f in enumerate(uploaded_files, start=1):
            img_label = f"Image {idx} ({f.name})"
            with st.spinner(f"Analyzing {f.name} using {scanner_engine}..."):
                img_bytes = f.getvalue()
                pil_img, books = process_bookshelf(img_bytes, idx, f"Image {idx}", scanner_engine, api_key)
                if pil_img is not None:
                    st.session_state.processed_images[img_label] = pil_img
                    st.session_state.master_books.extend(books)
        st.success(f"🎉 Analysis Complete! Detected {len(st.session_state.master_books)} books across {len(uploaded_files)} photos.")

if use_demo:
    if os.path.exists(SAMPLE_IMAGE):
        with open(SAMPLE_IMAGE, "rb") as f:
            demo_bytes = f.read()
        with st.spinner("Processing example bookstore shelf..."):
            pil_img, books = process_bookshelf(demo_bytes, 1, "Image 1 (Example Bookstore Shelf)", scanner_engine, api_key)
            st.session_state.processed_images = {"Image 1 (Example Bookstore Shelf)": pil_img}
            st.session_state.master_books = books
            st.success(f"🎉 Example shelf loaded: {len(books)} books identified!")

# Deduplication & Aggregation
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

# Main UI Display
if not display_catalog:
    st.info("👆 Tap 'Browse files' to upload photos from your phone, or tap 'Test with Example Shelf' to see a demonstration.")
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
                "Engine": b.get("source", "Hybrid")
            })
        st.dataframe(table_rows, width="stretch", height=620)

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Multi-Shelf AI • Hybrid Gemini + Offline")
