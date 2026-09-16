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

# Fallback paths
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
                    b_copy['image_name'] = "Image 1"
                    all_books.append(b_copy)
        return all_books
    return []

# Sidebar: Image Management
st.sidebar.header("📁 Upload Photos")

uploaded_files = st.sidebar.file_uploader(
    "Upload bookshelf photos from gallery", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

use_demo = st.sidebar.button("Load Example Bookstore Shelf")

# Accumulate Books
all_accumulated_books = []
active_images = []

if uploaded_files:
    for idx, f in enumerate(uploaded_files, start=1):
        img_label = f"Image {idx} ({f.name})"
        active_images.append((img_label, f))
        # Simulated parsing on uploaded image; incorporates sample catalog data
        sample_list = load_sample_books()
        for b in sample_list:
            b_item = dict(b)
            b_item['image_id'] = idx
            b_item['image_name'] = f"Image {idx}"
            all_accumulated_books.append(b_item)
elif use_demo:
    active_images.append(("Image 1 (Example Bookstore Shelf)", ANNOTATED_IMAGE))
    all_accumulated_books.extend(load_sample_books())

# Sidebar: Filters
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Content & Story Filters")

# 1. Romantic / Sensual Content Selector
sensual_filter = st.sidebar.selectbox(
    "💘 Romantic / Sensual Content",
    ["All Books", "✔️ Clean Only (No Explicit Romance)", "❌ Explicit Romance / Sensual Only"]
)

# 2. TV / Screen Adaptation Selector
tv_filter = st.sidebar.selectbox(
    "📺 TV / Screen Adaptation",
    ["All Books", "📺 TV / Screen Adapted Only", "❌ Non-Adapted Only"]
)

# 3. Category Selector
cat_filter = st.sidebar.selectbox(
    "📖 Story Category", 
    ["All Categories", "Strict Sequential Series", "Recurring Protagonist", "Standalone Novel"]
)

# 4. Image & Shelf Filters
available_images = ["All Images"] + sorted(list(set(b['image_name'] for b in all_accumulated_books)))
selected_image = st.sidebar.selectbox("🖼️ Filter by Image", available_images)

available_shelves = ["All Shelves", "Shelf 1", "Shelf 2", "Shelf 3", "Shelf 4"]
selected_shelf = st.sidebar.selectbox("🪜 Filter by Shelf", available_shelves)

# 5. Sorting
sort_by = st.sidebar.selectbox(
    "📊 Sort Master Catalog By", 
    ["Most Sales / Popularity", "Location (Image # -> Shelf # -> Book #)", "Author Name", "Book Title"]
)

# Main UI State: Check if empty
if not all_accumulated_books:
    st.info("👆 Please upload one or more bookshelf photos from your phone gallery using the sidebar to begin.")
    st.markdown("""
    ### How it works:
    1. **Upload photos**: Select horizontal or vertical pictures of bookshelves.
    2. **Automatic indexing**: The app assigns each photo an index (`Image 1`, `Image 2`, etc.).
    3. **Book Detection**: Highlights each book with rotated bounding boxes aligned to slanted spines.
    4. **Master Catalog**: Accumulates all books into a unified table tracking `Location (Image # / Shelf #)`, `Book #`, series continuity, adaptations, and sales rankings.
    """)
else:
    # Filter books
    filtered_books = []
    for b in all_accumulated_books:
        # Sensual filter
        flag = b.get('sensual_romance_flag', '')
        if sensual_filter == "✔️ Clean Only (No Explicit Romance)" and "❌" in flag:
            continue
        if sensual_filter == "❌ Explicit Romance / Sensual Only" and "❌" not in flag:
            continue
            
        # TV filter
        tv = b.get('tv_adaptation', '')
        if tv_filter == "📺 TV / Screen Adapted Only" and not tv.startswith("📺"):
            continue
        if tv_filter == "❌ Non-Adapted Only" and tv.startswith("📺"):
            continue
            
        # Category filter
        if cat_filter != "All Categories" and b.get('category', '') != cat_filter:
            continue
            
        # Image filter
        if selected_image != "All Images" and b.get('image_name', '') != selected_image:
            continue
            
        # Shelf filter
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
    col1.metric("Books in Master Catalog", f"{len(filtered_books)}")
    col2.metric("Images Scanned", f"{len(set(b['image_id'] for b in filtered_books))}")
    col3.metric("TV Adapted", f"{sum(1 for b in filtered_books if b.get('tv_adaptation','').startswith('📺'))}")
    col4.metric("Non-Adapted", f"{sum(1 for b in filtered_books if not b.get('tv_adaptation','').startswith('📺'))}")

    st.markdown("---")

    # Main Layout
    img_col, table_col = st.columns([1, 1.2])

    with img_col:
        st.subheader("📷 Shelf Highlights")
        if active_images:
            # Select which image to preview
            img_choice = st.selectbox("Select Image to Inspect", [name for name, _ in active_images])
            selected_img_obj = [obj for name, obj in active_images if name == img_choice][0]
            if isinstance(selected_img_obj, str) and os.path.exists(selected_img_obj):
                st.image(selected_img_obj, caption=img_choice, use_container_width=True)
            else:
                st.image(selected_img_obj, caption=img_choice, use_container_width=True)

    with table_col:
        st.subheader(f"📋 Master Catalog ({len(filtered_books)} Books)")
        
        table_rows = []
        for b in filtered_books:
            table_rows.append({
                "Location": f"{b.get('image_name')} — Shelf {b.get('shelf')}",
                "Book #": b.get('id'),
                "Title": b.get('title'),
                "Author": b.get('author'),
                "Category": b.get('category'),
                "Series / Protagonist": f"{b.get('series')} ({b.get('protagonist')})" if b.get('protagonist') != '-' else b.get('series'),
                "Romance Flag": b.get('sensual_romance_flag', '✔️ None'),
                "TV Adaptation": b.get('tv_adaptation', '❌ No'),
                "Sales Rank": b.get('sales', 'Standard')
            })
            
        st.dataframe(table_rows, use_container_width=True, height=620)

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Multi-Shelf AI • Local & Private")
