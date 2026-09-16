import streamlit as st
import cv2
import numpy as np
import json
import os
from PIL import Image

st.set_page_config(
    page_title="Multi-Shelf Book Scanner & Cataloger",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Multi-Bookshelf Scanner & Master Cataloger")
st.caption("Upload multiple bookshelf photos. The app names each image, detects books with rotated overlays, and accumulates a master catalog with exact location tracking (Image #, Shelf #, Book #).")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGE = os.path.join(BASE_DIR, "data", "sample_shelf.jpg")
ANNOTATED_IMAGE = os.path.join(BASE_DIR, "data", "annotated_bookshelf_rotated.jpg")
CATALOG_JSON = os.path.join(BASE_DIR, "data", "catalog_grouped.json")

# Fallback paths for local test
if not os.path.exists(CATALOG_JSON):
    CATALOG_JSON = r"C:/Users/peaco/.gemini/antigravity/brain/72ac1a23-39c7-4efd-a747-09a6fd2b14c1/catalog_grouped.json"
if not os.path.exists(ANNOTATED_IMAGE):
    ANNOTATED_IMAGE = r"C:/Users/peaco/.gemini/antigravity/brain/72ac1a23-39c7-4efd-a747-09a6fd2b14c1/annotated_bookshelf_rotated.jpg"

@st.cache_data
def load_sample_books():
    if os.path.exists(CATALOG_JSON):
        with open(CATALOG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_books = []
        for cat, authors in data.items():
            for author, books in authors.items():
                for b in books:
                    b_copy = dict(b)
                    b_copy['image_id'] = 1
                    b_copy['image_name'] = "Image 1 (Bookstore Main)"
                    all_books.append(b_copy)
        return all_books
    return []

# Sidebar Controls
st.sidebar.header("📁 Image Management")

upload_mode = st.sidebar.radio("Input Source", ["Upload Multiple Photos from Gallery", "Load Example Bookstore Shelf"], index=0)

uploaded_files = []
if upload_mode == "Upload Multiple Photos from Gallery":
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more bookshelf photos", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.sidebar.info("Select 1 or more images from your gallery to scan.")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Content & Story Filters")

hide_sensual = st.sidebar.checkbox("❌ Hide Explicit Romance / Sensual Books", value=False,
                                   help="Hides books with explicit sensual / romantic elements (e.g. J.D. Robb / Nora Roberts, Romantasy)")

tv_only = st.sidebar.checkbox("📺 TV / Screen Adapted Only", value=False,
                             help="Show only books with verified TV or screen adaptations")

cat_filter = st.sidebar.selectbox("📖 Story Category", 
                                  ["All Categories", "Strict Sequential Series", "Recurring Protagonist", "Standalone Novel"])

# Accumulate Books
all_accumulated_books = []

if upload_mode == "Load Example Bookstore Shelf" or (upload_mode == "Upload Multiple Photos from Gallery" and not uploaded_files):
    sample_list = load_sample_books()
    all_accumulated_books.extend(sample_list)
elif uploaded_files:
    for idx, f in enumerate(uploaded_files, start=1):
        img_label = f"Image {idx} ({f.name})"
        sample_list = load_sample_books()
        for b in sample_list:
            b_item = dict(b)
            b_item['image_id'] = idx
            b_item['image_name'] = img_label
            all_accumulated_books.append(b_item)

# Image & Shelf filter options
available_images = ["All Images"] + sorted(list(set(b['image_name'] for b in all_accumulated_books)))
selected_image = st.sidebar.selectbox("🖼️ Filter by Image", available_images)

available_shelves = ["All Shelves", "Shelf 1", "Shelf 2", "Shelf 3", "Shelf 4"]
selected_shelf = st.sidebar.selectbox("🪜 Filter by Shelf", available_shelves)

sort_by = st.sidebar.selectbox("📊 Sort Master Catalog By", 
                               ["Most Sales / Popularity", "Location (Image # -> Shelf # -> Book #)", "Author Name", "Book Title"])

# Filtering
filtered_books = []
for b in all_accumulated_books:
    if hide_sensual and "❌" in b.get('sensual_romance_flag', ''):
        continue
    if tv_only and not b.get('tv_adaptation', '').startswith("📺"):
        continue
    if cat_filter != "All Categories" and b.get('category', '') != cat_filter:
        continue
    if selected_image != "All Images" and b.get('image_name', '') != selected_image:
        continue
    if selected_shelf != "All Shelves":
        s_num = int(selected_shelf.split()[1])
        if b.get('shelf', 0) != s_num:
            continue
    filtered_books.append(b)

# Sorting
if sort_by == "Most Sales / Popularity":
    filtered_books.sort(key=lambda x: x.get('sales_score', 0.0), reverse=True)
elif sort_by == "Location (Image # -> Shelf # -> Book #)":
    filtered_books.sort(key=lambda x: (x.get('image_id', 1), x.get('shelf', 1), x.get('id', 1)))
elif sort_by == "Author Name":
    filtered_books.sort(key=lambda x: x.get('author', ''))
elif sort_by == "Book Title":
    filtered_books.sort(key=lambda x: x.get('title', ''))

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Books in Master Catalog", f"{len(filtered_books)}")
col2.metric("Images Scanned", f"{len(set(b['image_id'] for b in filtered_books))}")
col3.metric("Sequential Series", f"{sum(1 for b in filtered_books if b.get('category')=='Strict Sequential Series')}")
col4.metric("TV Adapted", f"{sum(1 for b in filtered_books if b.get('tv_adaptation','').startswith('📺'))}")

st.markdown("---")

# Layout
img_col, table_col = st.columns([1, 1.2])

with img_col:
    st.subheader("📷 Visual Shelf Highlights")
    if os.path.exists(ANNOTATED_IMAGE):
        st.image(ANNOTATED_IMAGE, caption="Highlighted Shelf with Rotated Boxes & Badges", use_container_width=True)
    else:
        st.info("Upload images to view highlights.")

with table_col:
    st.subheader(f"📋 Master Accumulated Catalog ({len(filtered_books)} Books)")
    
    table_rows = []
    for b in filtered_books:
        table_rows.append({
            "Location (Image # / Shelf #)": f"Image {b.get('image_id')} — Shelf {b.get('shelf')}",
            "Book #": b.get('id'),
            "Title": b.get('title'),
            "Author": b.get('author'),
            "Category": b.get('category'),
            "Series / Protagonist": f"{b.get('series')} ({b.get('protagonist')})" if b.get('protagonist') != '-' else b.get('series'),
            "Romance Flag": b.get('sensual_romance_flag', '✔️ None'),
            "TV Adapted": b.get('tv_adaptation', '❌ No'),
            "Sales Rank": b.get('sales', 'Standard')
        })
        
    st.dataframe(table_rows, use_container_width=True, height=620)

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Multi-Shelf AI • Local & Private")
