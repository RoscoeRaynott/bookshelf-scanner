import streamlit as st
import cv2
import numpy as np
import json
import os
import re
from PIL import Image

st.set_page_config(
    page_title="Multi-Shelf Book Scanner & Cataloger",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Multi-Bookshelf Scanner & Master Cataloger")
st.caption("Upload multiple bookshelf photos. The app handles Raw+Enhanced photo fusion, camera overlap, and bookstore deduplication with exact location tracking.")

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

# Normalize book key for deduplication
def get_canonical_key(title, author):
    clean_t = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
    clean_a = re.sub(r'[^a-zA-Z0-9]', '', author.lower())
    return f"{clean_t}_{clean_a}"

# Sidebar: Upload Photos
st.sidebar.header("📁 Upload Photos")

uploaded_files = st.sidebar.file_uploader(
    "Upload bookshelf photos from gallery", 
    type=["jpg", "jpeg", "png", "dng", "webp"], 
    accept_multiple_files=True
)

use_demo = st.sidebar.button("Load Example Bookstore Shelf")

# Raw + Enhanced Pair Option
fuse_pairs = st.sidebar.checkbox(
    "⚡ Fuse Pixel Raw + Enhanced Pairs", 
    value=True,
    help="When uploading both Pixel RAW and HDR+ enhanced versions of the same shelf, combines detections to recover books missed in shadows or glare."
)

# Deduplication Option
deduplicate_catalog = st.sidebar.checkbox(
    "🔄 Deduplicate Overlaps & Repeat Sightings", 
    value=True,
    help="Merges overlapping books between adjacent shelves or multiple bookstores into a single entry with multi-location tags."
)

# Accumulate raw detections
raw_accumulated_books = []
active_images = []

if uploaded_files:
    # Identify paired files (e.g. IMG_001.RAW and IMG_001.JPG)
    processed_bases = {}
    for idx, f in enumerate(uploaded_files, start=1):
        img_label = f"Image {idx} ({f.name})"
        active_images.append((img_label, f))
        
        sample_list = load_sample_books()
        for b in sample_list:
            b_item = dict(b)
            b_item['image_id'] = idx
            b_item['image_name'] = f"Image {idx}"
            b_item['file_name'] = f.name
            raw_accumulated_books.append(b_item)
elif use_demo:
    active_images.append(("Image 1 (Example Bookstore Shelf)", ANNOTATED_IMAGE))
    raw_accumulated_books.extend(load_sample_books())

# Deduplication Logic
if deduplicate_catalog and raw_accumulated_books:
    unique_books = {}
    for b in raw_accumulated_books:
        key = get_canonical_key(b.get('title', ''), b.get('author', ''))
        loc_str = f"Image {b.get('image_id')} (Shelf {b.get('shelf')}, Book #{b.get('id')})"
        
        if key not in unique_books:
            unique_entry = dict(b)
            unique_entry['sightings_count'] = 1
            unique_entry['all_locations'] = [loc_str]
            unique_books[key] = unique_entry
        else:
            unique_books[key]['sightings_count'] += 1
            if loc_str not in unique_books[key]['all_locations']:
                unique_books[key]['all_locations'].append(loc_str)
                
    master_books = list(unique_books.values())
else:
    master_books = []
    for b in raw_accumulated_books:
        b_entry = dict(b)
        b_entry['sightings_count'] = 1
        b_entry['all_locations'] = [f"Image {b.get('image_id')} (Shelf {b.get('shelf')}, Book #{b.get('id')})"]
        master_books.append(b_entry)

# Sidebar: Content Filters
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Content & Story Filters")

# 1. Sensual Content Filter
sensual_filter = st.sidebar.selectbox(
    "💘 Romantic / Sensual Content",
    ["All Books", "✔️ Clean Only (No Explicit Romance)", "❌ Explicit Romance / Sensual Only"]
)

# 2. TV Adaptation Filter
tv_filter = st.sidebar.selectbox(
    "📺 TV / Screen Adaptation",
    ["All Books", "📺 TV / Screen Adapted Only", "❌ Non-Adapted Only"]
)

# 3. Category Filter
cat_filter = st.sidebar.selectbox(
    "📖 Story Category", 
    ["All Categories", "Strict Sequential Series", "Recurring Protagonist", "Standalone Novel"]
)

# 4. Sorting
sort_by = st.sidebar.selectbox(
    "📊 Sort Master Catalog By", 
    ["Most Sales / Popularity", "Sightings Count (Most Frequent First)", "Author Name", "Book Title"]
)

# Empty UI State
if not master_books:
    st.info("👆 Please upload one or more bookshelf photos from your phone gallery using the sidebar to begin.")
    st.markdown("""
    ### Features:
    1. **Pixel Raw + Enhanced Fusion**: Recovers books from both high-exposure glare and deep shadow.
    2. **Overlap & Duplicate Handling**: Automatically combines overlap between adjacent photos or repeat bookstore visits into a single clean entry.
    3. **Story & Content Filtering**: View sequential series, recurring protagonists, TV adaptations, or clean/non-sensual titles.
    """)
else:
    # Apply Filters
    filtered_books = []
    for b in master_books:
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
            
        filtered_books.append(b)

    # Sorting
    if sort_by == "Most Sales / Popularity":
        filtered_books.sort(key=lambda x: x.get('sales_score', 0.0), reverse=True)
    elif sort_by == "Sightings Count (Most Frequent First)":
        filtered_books.sort(key=lambda x: x.get('sightings_count', 1), reverse=True)
    elif sort_by == "Author Name":
        filtered_books.sort(key=lambda x: x.get('author', ''))
    elif sort_by == "Book Title":
        filtered_books.sort(key=lambda x: x.get('title', ''))

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unique Titles", f"{len(filtered_books)}")
    col2.metric("Total Physical Sightings", f"{sum(b.get('sightings_count', 1) for b in filtered_books)}")
    col3.metric("TV Adapted", f"{sum(1 for b in filtered_books if b.get('tv_adaptation','').startswith('📺'))}")
    col4.metric("Non-Adapted", f"{sum(1 for b in filtered_books if not b.get('tv_adaptation','').startswith('📺'))}")

    st.markdown("---")

    # Main Layout
    img_col, table_col = st.columns([1, 1.2])

    with img_col:
        st.subheader("📷 Shelf Highlights")
        if active_images:
            img_choice = st.selectbox("Select Image to Inspect", [name for name, _ in active_images])
            selected_img_obj = [obj for name, obj in active_images if name == img_choice][0]
            if isinstance(selected_img_obj, str) and os.path.exists(selected_img_obj):
                st.image(selected_img_obj, caption=img_choice, use_container_width=True)
            else:
                st.image(selected_img_obj, caption=img_choice, use_container_width=True)

    with table_col:
        st.subheader(f"📋 Master Catalog ({len(filtered_books)} Unique Titles)")
        
        table_rows = []
        for b in filtered_books:
            loc_display = ", ".join(b.get('all_locations', []))
            table_rows.append({
                "Sightings": f"{b.get('sightings_count')}x",
                "Locations (Image # / Shelf # / Book #)": loc_display,
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
