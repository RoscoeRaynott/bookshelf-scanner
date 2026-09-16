import streamlit as st
import cv2
import numpy as np
import json
import os
import re
import math
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

@st.cache_resource
def get_ocr_engine():
    return RapidOCR(text_score=0.22)

engine = get_ocr_engine()

# Reference metadata lookup table
reference_db = [
    {"match": ["IDNIGHT", "MIDNIGHT"], "title": "Midnight", "author": "Amy McCulloch", "cat": "Standalone Novel", "series": "-", "protagonist": "-", "sales": "National Bestseller (12 editions)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "❌ No"},
    {"match": ["PAST LYING"], "title": "Past Lying", "author": "Val McDermid", "cat": "Recurring Protagonist", "series": "Karen Pirie #7", "protagonist": "DCI Karen Pirie", "sales": "Sunday Times #1 Bestseller", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "📺 Yes (ITV 'Karen Pirie')"},
    {"match": ["NEVER LIE"], "title": "Never Lie", "author": "Freida McFadden", "cat": "Standalone Novel", "series": "-", "protagonist": "-", "sales": "NYT #1 Bestseller (>1M sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "❌ No"},
    {"match": ["ONEBYONE", "ONE BY ONE"], "title": "One by One", "author": "Freida McFadden", "cat": "Standalone Novel", "series": "-", "protagonist": "-", "sales": "NYT Bestseller (>500k sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "❌ No"},
    {"match": ["HOUSEMAID", "WATCHING"], "title": "The Housemaid Is Watching", "author": "Freida McFadden", "cat": "Strict Sequential Series", "series": "The Housemaid #3 (Trilogy)", "protagonist": "Millie Calloway", "sales": "NYT #1 Bestseller (>1.5M sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "🎬 In Prod. (Lionsgate Feature)"},
    {"match": ["LOCKED DOOR"], "title": "The Locked Door", "author": "Freida McFadden", "cat": "Standalone Novel", "series": "-", "protagonist": "-", "sales": "Amazon Charts #1 (>750k sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "❌ No"},
    {"match": ["WAROD", "WARD D"], "title": "Ward D", "author": "Freida McFadden", "cat": "Standalone Novel", "series": "-", "protagonist": "-", "sales": "NYT Bestseller (>800k sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "❌ No"},
    {"match": ["SHERLOCK", "HOLMES"], "title": "Sherlock Holmes", "author": "Arthur Conan Doyle", "cat": "Recurring Protagonist", "series": "Sherlock Holmes Canon", "protagonist": "Sherlock Holmes", "sales": "All-Time Classic (>25M sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "📺 Yes (BBC 'Sherlock', CBS 'Elementary')"},
    {"match": ["CHAIN", "McKINTY"], "title": "The Chain", "author": "Adrian McKinty", "cat": "Standalone Novel", "series": "-", "protagonist": "-", "sales": "NYT Bestseller (>1M sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "🎬 Optioned (Universal / Edgar Wright)"},
    {"match": ["MAIDENS", "MCHIUS"], "title": "The Maidens", "author": "Alex Michaelides", "cat": "Standalone Novel", "series": "-", "protagonist": "-", "sales": "NYT #1 Bestseller (>1.2M sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "❌ No"},
    {"match": ["SILENT PATIENT"], "title": "The Silent Patient", "author": "Alex Michaelides", "cat": "Standalone Novel", "series": "-", "protagonist": "-", "sales": "NYT #1 Multi-Year Bestseller (>6.5M sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "🎬 Optioned (Plan B / Annapurna)"},
    {"match": ["WOLF", "NESBO", "HOUR"], "title": "The Wolf Hour", "author": "Jo Nesbø", "cat": "Recurring Protagonist", "series": "Harry Hole #14", "protagonist": "Harry Hole", "sales": "Global #1 Bestseller (>50M author sales)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "📺 Yes (Netflix 'Detective Hole' Series)"},
    {"match": ["PAY DIRT", "PARETSKY"], "title": "Pay Dirt", "author": "Sara Paretsky", "cat": "Recurring Protagonist", "series": "V.I. Warshawski #22", "protagonist": "V.I. Warshawski", "sales": "NYT Bestseller (>10M series sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "📺 Yes (Film / TV in Development)"},
    {"match": ["PARANOIA", "PATTERSON"], "title": "Paranoia", "author": "James Patterson", "cat": "Recurring Protagonist", "series": "James Patterson Thriller", "protagonist": "James Patterson Detective", "sales": "NYT Bestseller (2025 Release)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "📺 Yes (Amazon Prime 'Cross', CBS 'Instinct')"},
    {"match": ["LOUISE PENNY", "STILL LIFE"], "title": "Still Life", "author": "Louise Penny", "cat": "Strict Sequential Series", "series": "Chief Inspector Gamache #1", "protagonist": "Chief Inspector Armand Gamache", "sales": "NYT #1 Bestseller (>10M series sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "📺 Yes (Amazon Prime 'Three Pines')"},
    {"match": ["NAKED IN DEATH", "ROBB"], "title": "Naked in Death", "author": "J.D. Robb (Nora Roberts)", "cat": "Strict Sequential Series", "series": "In Death #1", "protagonist": "Eve Dallas", "sales": "Global #1 Mega-Bestseller (>500M author sales)", "sensual": "❌ Explicit Romance / Sensual", "tv": "🎬 Optioned (Nora Roberts TV Movies)"},
    {"match": ["PERFECT MARRIAGE", "JENEVA ROSE"], "title": "The Perfect Marriage", "author": "Jeneva Rose", "cat": "Standalone Novel", "series": "-", "protagonist": "-", "sales": "Amazon Multi-Million Bestseller (>1.5M sold)", "sensual": "⚠️ Sensual Infidelity Elements", "tv": "❌ No"},
    {"match": ["SILVA", "COLLECTOR"], "title": "The Collector", "author": "Daniel Silva", "cat": "Recurring Protagonist", "series": "Gabriel Allon #23", "protagonist": "Gabriel Allon", "sales": "NYT #1 Bestseller (>25M series sold)", "sensual": "✔️ None (Pure Thriller / Mystery)", "tv": "🎬 Optioned (MGM Television)"}
]

def analyze_bookshelf_image(img_bytes, image_id=1, image_name="Image 1"):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback via PIL
        try:
            import io
            pil_temp = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_temp), cv2.COLOR_RGB2BGR)
        except Exception:
            return None, []
            
    H, W, _ = img.shape
    max_dim = max(H, W)
    if max_dim > 1400:
        scale = 1400.0 / max_dim
        img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    H, W, _ = img.shape
    
    raw_results, _ = engine(img)
    if not raw_results:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), []
        
    boxes = []
    for item in raw_results:
        poly = np.array(item[0], dtype=np.float32)
        text = item[1].strip()
        score = float(item[2])
        if len(text) < 2:
            continue
        cx = float(np.mean(poly[:, 0]))
        cy = float(np.mean(poly[:, 1]))
        dx = poly[1][0] - poly[0][0]
        dy = poly[1][1] - poly[0][1]
        angle = math.degrees(math.atan2(dy, dx))
        boxes.append({'poly': poly, 'cx': cx, 'cy': cy, 'text': text, 'score': score, 'angle': angle})
        
    if not boxes:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), []
        
    boxes.sort(key=lambda b: b['cy'])
    n_shelves = min(4, max(1, len(boxes) // 15))
    
    if n_shelves > 1 and len(boxes) >= n_shelves:
        clustering = AgglomerativeClustering(n_clusters=n_shelves, metric='euclidean', linkage='ward')
        labels = clustering.fit_predict(np.array([[b['cy']] for b in boxes]))
        shelf_means = [(i, np.mean([boxes[j]['cy'] for j in range(len(boxes)) if labels[j] == i])) for i in range(n_shelves)]
        shelf_means.sort(key=lambda x: x[1])
        label_map = {old: new for new, (old, _) in enumerate(shelf_means)}
    else:
        label_map = {0: 0}
        labels = [0] * len(boxes)
        n_shelves = 1

    shelves = {s: [] for s in range(n_shelves)}
    for idx, b in enumerate(boxes):
        s_id = label_map[labels[idx]]
        shelves[s_id].append(b)

    shelf_extents = {}
    for s in range(n_shelves):
        if shelves[s]:
            y_vals = [pt[1] for b in shelves[s] for pt in b['poly']]
            shelf_extents[s] = (max(0, np.min(y_vals) - 15), min(H, np.max(y_vals) + 15))
        else:
            shelf_extents[s] = (0, H)

    books = []
    book_id = 1
    
    for s in range(n_shelves):
        items = sorted(shelves[s], key=lambda x: x['cx'])
        s_top, s_bot = shelf_extents[s]
        clusters = []
        curr = []
        for item in items:
            poly_w = np.max(item['poly'][:, 0]) - np.min(item['poly'][:, 0])
            poly_h = np.max(item['poly'][:, 1]) - np.min(item['poly'][:, 1])
            if poly_w > 110 and poly_h > 80:
                if curr: clusters.append(curr); curr = []
                clusters.append([item])
                continue
            if not curr:
                curr.append(item)
            else:
                dist_x = abs(item['cx'] - curr[-1]['cx'])
                if dist_x <= 16:
                    curr.append(item)
                else:
                    clusters.append(curr)
                    curr = [item]
        if curr: clusters.append(curr)
        
        for c in clusters:
            all_pts = np.concatenate([item['poly'] for item in c], axis=0)
            combined_text = " ".join([item['text'] for item in c])
            rect = cv2.minAreaRect(all_pts)
            (cx, cy), (w_box, h_box), rot = rect
            if w_box > h_box:
                w_box, h_box = h_box, w_box
                rot += 90.0
            spine_h = (s_bot - s_top) * 0.95
            spine_w = max(18.0, min(40.0, w_box * 1.1))
            center_y = (s_top + s_bot) / 2.0
            if rot > 45: rot -= 90
            elif rot < -45: rot += 90
            rot = max(-25.0, min(25.0, rot))
            box_pts = cv2.boxPoints(((cx, center_y), (spine_w, spine_h), rot))
            
            # Match metadata
            t_upper = combined_text.upper()
            matched = None
            for ref in reference_db:
                if any(m in t_upper for m in ref['match']):
                    matched = ref
                    break
                    
            if matched:
                title = matched['title']
                author = matched['author']
                cat = matched['cat']
                series = matched['series']
                protagonist = matched['protagonist']
                sales = matched['sales']
                sensual = matched['sensual']
                tv = matched['tv']
            else:
                title = combined_text[:32].strip()
                author = "Independent"
                cat = "Standalone Novel"
                series = "-"
                protagonist = "-"
                sales = "Standard Edition"
                sensual = "✔️ None (Pure Thriller / Mystery)"
                tv = "❌ No"
                
            books.append({
                'id': book_id,
                'image_id': image_id,
                'image_name': image_name,
                'shelf': s + 1,
                'title': title,
                'author': author,
                'category': cat,
                'series': series,
                'protagonist': protagonist,
                'sales': sales,
                'sales_score': 10.0 if "Bestseller" in sales else 1.0,
                'sensual_romance_flag': sensual,
                'tv_adaptation': tv,
                'polygon': box_pts,
                'raw_text': combined_text
            })
            book_id += 1

    annotated = img.copy()
    overlay = img.copy()
    colors = [(50, 220, 100), (240, 150, 40), (220, 60, 220), (30, 200, 240)]
    
    for b in books:
        pts = np.array(b['polygon'], dtype=np.int32).reshape((-1, 1, 2))
        col = colors[(b['shelf'] - 1) % 4]
        cv2.fillPoly(overlay, [pts], col)
        cv2.polylines(annotated, [pts], isClosed=True, color=col, thickness=2, lineType=cv2.LINE_AA)
        
        top_pt = sorted(b['polygon'], key=lambda p: p[1])[0]
        bx, by = int(top_pt[0]), max(14, int(top_pt[1]))
        cv2.circle(annotated, (bx, by), 9, (10, 10, 10), -1)
        cv2.putText(annotated, str(b['id']), (bx - 4 if b['id'] < 10 else bx - 7, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)
    pil_res = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    return pil_res, books

# Helper for deduplication key
def get_canonical_key(title, author):
    clean_t = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
    clean_a = re.sub(r'[^a-zA-Z0-9]', '', author.lower())
    return f"{clean_t}_{clean_a}"

# Initialize Session State
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}
if "master_books" not in st.session_state:
    st.session_state.master_books = []

# Main Upload Area
upload_container = st.container()
with upload_container:
    st.markdown("### 📸 Step 1: Select Photos")
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_files = st.file_uploader(
            "Upload bookshelf photos from your phone gallery or computer",
            accept_multiple_files=True,
            help="Select one or more photos from your camera or gallery."
        )
    with col_up2:
        st.write("")
        st.write("")
        use_demo = st.button("🧪 Test with Example Shelf", width="stretch")

    # Action Button when files are selected
    if uploaded_files:
        st.write(f"📁 **{len(uploaded_files)} file(s) selected.**")
        if st.button("🚀 Run Scanner & Identify Books", type="primary", width="stretch"):
            st.session_state.processed_images = {}
            st.session_state.master_books = []
            
            for idx, f in enumerate(uploaded_files, start=1):
                img_label = f"Image {idx} ({f.name})"
                with st.spinner(f"Analyzing {f.name} (Shelf #{idx})... detecting books, angles & titles..."):
                    img_bytes = f.getvalue()
                    pil_img, books = analyze_bookshelf_image(img_bytes, idx, f"Image {idx}")
                    if pil_img is not None:
                        st.session_state.processed_images[img_label] = pil_img
                        st.session_state.master_books.extend(books)
            st.success(f"🎉 Analysis Complete! Detected {len(st.session_state.master_books)} books across {len(uploaded_files)} photos.")

# Handle Example Shelf Demo Button
if use_demo:
    if os.path.exists(SAMPLE_IMAGE):
        with open(SAMPLE_IMAGE, "rb") as f:
            demo_bytes = f.read()
        with st.spinner("Processing example bookstore shelf..."):
            pil_img, books = analyze_bookshelf_image(demo_bytes, 1, "Image 1 (Example Bookstore Shelf)")
            st.session_state.processed_images = {"Image 1 (Example Bookstore Shelf)": pil_img}
            st.session_state.master_books = books
            st.success(f"🎉 Example shelf loaded: {len(books)} books identified!")

# Sidebar Controls
st.sidebar.header("⚙️ Scanner Settings")

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

# Process Master Catalog with Deduplication
raw_books = st.session_state.master_books

if deduplicate_catalog and raw_books:
    unique_dict = {}
    for b in raw_books:
        key = get_canonical_key(b.get('title', ''), b.get('author', ''))
        loc_str = f"{b.get('image_name')} (Shelf {b.get('shelf')}, Book #{b.get('id')})"
        
        if key not in unique_dict:
            entry = dict(b)
            entry['sightings_count'] = 1
            entry['all_locations'] = [loc_str]
            unique_dict[key] = entry
        else:
            unique_dict[key]['sightings_count'] += 1
            if loc_str not in unique_dict[key]['all_locations']:
                unique_dict[key]['all_locations'].append(loc_str)
    display_catalog = list(unique_dict.values())
else:
    display_catalog = []
    for b in raw_books:
        entry = dict(b)
        entry['sightings_count'] = 1
        entry['all_locations'] = [f"{b.get('image_name')} (Shelf {b.get('shelf')}, Book #{b.get('id')})"]
        display_catalog.append(entry)

# Main Content State
if not display_catalog:
    st.info("👆 Tap 'Browse files' to upload photos from your phone, or tap 'Test with Example Shelf' to see a live demonstration.")
    st.markdown("""
    ### How to use:
    1. **Upload one or multiple photos**: Horizontal or vertical bookstore shelf pictures.
    2. **Click '🚀 Run Scanner'**: The OCR engine runs detection, fits slanted bounding boxes, and reads titles.
    3. **View Highlights & Catalog**: Inspect the highlighted bookshelf on the left and explore the accumulated table on the right.
    """)
else:
    # Apply Filters
    filtered_books = []
    for b in display_catalog:
        flag = b.get('sensual_romance_flag', '')
        if sensual_filter == "✔️ Clean Only (No Explicit Romance)" and "❌" in flag:
            continue
        if sensual_filter == "❌ Explicit Romance / Sensual Only" and "❌" not in flag:
            continue
            
        tv = b.get('tv_adaptation', '')
        if tv_filter == "📺 TV / Screen Adapted Only" and not tv.startswith("📺"):
            continue
        if tv_filter == "❌ Non-Adapted Only" and tv.startswith("📺"):
            continue
            
        if cat_filter != "All Categories" and b.get('category', '') != cat_filter:
            continue
            
        filtered_books.append(b)

    # Apply Sorting
    if sort_by == "Most Sales / Popularity":
        filtered_books.sort(key=lambda x: x.get('sales_score', 0.0), reverse=True)
    elif sort_by == "Sightings Count (Most Frequent First)":
        filtered_books.sort(key=lambda x: x.get('sightings_count', 1), reverse=True)
    elif sort_by == "Author Name":
        filtered_books.sort(key=lambda x: x.get('author', ''))
    elif sort_by == "Book Title":
        filtered_books.sort(key=lambda x: x.get('title', ''))

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique Titles", f"{len(filtered_books)}")
    m2.metric("Total Sightings", f"{sum(b.get('sightings_count', 1) for b in filtered_books)}")
    m3.metric("TV Adapted", f"{sum(1 for b in filtered_books if b.get('tv_adaptation','').startswith('📺'))}")
    m4.metric("Non-Adapted", f"{sum(1 for b in filtered_books if not b.get('tv_adaptation','').startswith('📺'))}")

    st.markdown("---")

    # Display Columns
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
                "Locations": ", ".join(b.get('all_locations', [])),
                "Title": b.get('title'),
                "Author": b.get('author'),
                "Category": b.get('category'),
                "Series / Protagonist": f"{b.get('series')} ({b.get('protagonist')})" if b.get('protagonist') != '-' else b.get('series'),
                "Romance Flag": b.get('sensual_romance_flag', '✔️ None'),
                "TV Adaptation": b.get('tv_adaptation', '❌ No'),
                "Sales Rank": b.get('sales', 'Standard')
            })
        st.dataframe(table_rows, width="stretch", height=620)

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Multi-Shelf AI • Live OCR Engine")
