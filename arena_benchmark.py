import os
import re
import json
import time
import urllib.request
import urllib.parse

ARENA_25_BOOKS = [
    # Tier 1: Mega-Bestsellers
    {"id": 1, "title": "The Silent Patient", "author": "Alex Michaelides", "tier": "Mega-Bestseller"},
    {"id": 2, "title": "The Housemaid", "author": "Freida McFadden", "tier": "Mega-Bestseller"},
    {"id": 3, "title": "Along Came a Spider", "author": "James Patterson", "tier": "Mega-Bestseller"},
    {"id": 4, "title": "Naked in Death", "author": "J.D. Robb", "tier": "Mega-Bestseller"},
    {"id": 5, "title": "The Snowman", "author": "Jo Nesbø", "tier": "Mega-Bestseller"},
    {"id": 6, "title": "The Kill Artist", "author": "Daniel Silva", "tier": "Mega-Bestseller"},
    
    # Tier 2: Popular Bestsellers
    {"id": 7, "title": "The Maid", "author": "Nita Prose", "tier": "Popular Bestseller"},
    {"id": 8, "title": "The Guest List", "author": "Lucy Foley", "tier": "Popular Bestseller"},
    {"id": 9, "title": "The Whisper Man", "author": "Alex North", "tier": "Popular Bestseller"},
    {"id": 10, "title": "None of This Is True", "author": "Lisa Jewell", "tier": "Popular Bestseller"},
    {"id": 11, "title": "The Only One Left", "author": "Riley Sager", "tier": "Popular Bestseller"},
    {"id": 12, "title": "The Perfect Marriage", "author": "Jeneva Rose", "tier": "Popular Bestseller"},
    
    # Tier 3: Award Winners & Notable Series
    {"id": 13, "title": "The Chain", "author": "Adrian McKinty", "tier": "Award Winner / Series"},
    {"id": 14, "title": "Devil in a Blue Dress", "author": "Walter Mosley", "tier": "Award Winner / Series"},
    {"id": 15, "title": "The Mermaids Singing", "author": "Val McDermid", "tier": "Award Winner / Series"},
    {"id": 16, "title": "Secret Identity", "author": "Alex Segura", "tier": "Award Winner / Series"},
    {"id": 17, "title": "A Line to Kill", "author": "Anthony Horowitz", "tier": "Award Winner / Series"},
    {"id": 18, "title": "Lavender House", "author": "Lev AC Rosen", "tier": "Award Winner / Series"},
    {"id": 19, "title": "The Butcher's Boy", "author": "Thomas Perry", "tier": "Award Winner / Series"},
    
    # Tier 4: Midlist & Contemporary Mystery
    {"id": 20, "title": "A Trace of Deceit", "author": "Karen Odden", "tier": "Midlist / Mystery"},
    {"id": 21, "title": "The Verifiers", "author": "Jane Pek", "tier": "Midlist / Mystery"},
    {"id": 22, "title": "The Overnight Guest", "author": "Heather Gudenkauf", "tier": "Midlist / Mystery"},
    {"id": 23, "title": "Local Woman Missing", "author": "Mary Kubica", "tier": "Midlist / Mystery"},
    {"id": 24, "title": "The Quiet Tenant", "author": "Clémence Michallon", "tier": "Midlist / Mystery"},
    {"id": 25, "title": "The Maidens", "author": "Alex Michaelides", "tier": "Midlist / Mystery"}
]


def query_free_books_api(title, author, google_key=None):
    """Query Google Books API (with Open Library fallback). 100% free database lookup."""
    t0 = time.time()
    q = f"intitle:{title}+inauthor:{author}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={urllib.parse.quote(q)}&maxResults=1"
    # AQ keys are Google AI Studio keys, not Google Books API keys. Only attach if non-AQ key.
    if google_key and not str(google_key).strip().startswith("AQ"):
        url += f"&key={google_key}"
    
    req = urllib.request.Request(url, headers={"User-Agent": "BookshelfScanner/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("items", [])
            if items:
                v = items[0].get("volumeInfo", {})
                latency = round((time.time() - t0) * 1000, 1)
                categories = v.get("categories", ["Fiction"])
                return {
                    "method": "Google Books API",
                    "title": v.get("title", title),
                    "author": ", ".join(v.get("authors", [author])),
                    "publisher": v.get("publisher", "-"),
                    "published_year": str(v.get("publishedDate", "-"))[:4],
                    "page_count": v.get("pageCount", "-"),
                    "category": categories[0] if categories else "General Fiction",
                    "description": (v.get("description") or "")[:250] + ("…" if v.get("description") and len(v.get("description")) > 250 else ""),
                    "sales_reported": "Not reported in catalog databases",
                    "tv_deal": "Not indexed in catalog databases",
                    "cost_usd": 0.0,
                    "latency_ms": latency,
                    "status": "Success"
                }
    except Exception:
        pass

    # Fallback to Open Library (Public domain, zero key required)
    try:
        ol_url = f"https://openlibrary.org/search.json?title={urllib.parse.quote(title)}&author={urllib.parse.quote(author)}&limit=1"
        ol_req = urllib.request.Request(ol_url, headers={"User-Agent": "BookshelfScanner/2.0 (contact@bookshelf-scanner.app)"})
        with urllib.request.urlopen(ol_req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            docs = data.get("docs", [])
            if docs:
                d = docs[0]
                latency = round((time.time() - t0) * 1000, 1)
                return {
                    "method": "Open Library API",
                    "title": d.get("title", title),
                    "author": ", ".join(d.get("author_name", [author])),
                    "publisher": d.get("publisher", ["-"])[0] if d.get("publisher") else "-",
                    "published_year": str(d.get("first_publish_year", "-")),
                    "page_count": d.get("number_of_pages_median", "-"),
                    "category": (d.get("subject", ["Fiction"]) or ["Fiction"])[0],
                    "description": "Retrieved from Open Library open database",
                    "sales_reported": "Not reported in catalog databases",
                    "tv_deal": "Not indexed in catalog databases",
                    "cost_usd": 0.0,
                    "latency_ms": latency,
                    "status": "Success"
                }
    except Exception as ex:
        return {
            "method": "Books API (Error)",
            "title": title,
            "author": author,
            "status": f"Lookup failed: {ex}",
            "cost_usd": 0.0,
            "latency_ms": round((time.time() - t0) * 1000, 1)
        }

    return {
        "method": "Books API (Not Found)",
        "title": title,
        "author": author,
        "status": "No record found in Google Books or Open Library",
        "cost_usd": 0.0,
        "latency_ms": round((time.time() - t0) * 1000, 1)
    }


def query_direct_gemini_api(title, author, gemini_api_key):
    """Query Google AI Studio Gemini Free Tier API ($0.00 up to 1,500 calls/day)."""
    if not gemini_api_key:
        return {
            "method": "Google AI Studio",
            "title": title,
            "author": author,
            "status": "Error: GEMINI_API_KEY is not set",
            "cost_usd": 0.0,
            "latency_ms": 0.0
        }

    clean_key = str(gemini_api_key).strip().strip("\"'")
    t0 = time.time()
    
    prompt = f"""You are an objective book industry research agent.
Analyze the published book '{title}' by author '{author}'.
Extract and return strictly a valid JSON object with these keys:
- "book_sales": Verified copy count across all formats (print, ebook, audio) if publicly reported, otherwise strictly 'Not publicly reported'.
- "author_fame": Author's verified lifetime career sales if publicly reported, otherwise strictly 'Not publicly reported'.
- "author_fame_score": Integer total lifetime copies sold (e.g. 50000000 for 50M, 0 if unknown).
- "tv_adaptation": Film/TV adaptation status: 'Yes (details)', 'Optioned', or 'No'.
- "sensual_rating": 'Explicit Romance', 'Moderate Romance', or 'Clean / None'.
- "category": Book genre (e.g. 'Psychological Thriller', 'Domestic Suspense', 'Police Procedural', etc.).
- "series": Series name or 'Standalone Novel'.
- "protagonist": Lead recurring character or '-'.
- "evidence": 1-sentence source summary.

Output ONLY the JSON object, no commentary."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }
    payload_bytes = json.dumps(payload).encode("utf-8")

    # New standard: Pass API key via x-goog-api-key header (required for AQ. Authentication Keys)
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": clean_key
    }

    # Candidate models in order
    candidate_models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
    last_error = ""

    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        req = urllib.request.Request(url, data=payload_bytes, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                raw_body = json.loads(resp.read().decode("utf-8"))
                candidates = raw_body.get("candidates", [])
                if candidates:
                    part_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                    parsed = json.loads(part_text)
                    latency = round((time.time() - t0) * 1000, 1)
                    return {
                        "method": f"Google AI Studio ({model})",
                        "title": title,
                        "author": author,
                        "book_sales": parsed.get("book_sales", "Not publicly reported"),
                        "author_fame": parsed.get("author_fame", "Not publicly reported"),
                        "author_fame_score": parsed.get("author_fame_score", 0),
                        "tv_deal": parsed.get("tv_adaptation", "No"),
                        "sensual_rating": parsed.get("sensual_rating", "Clean / None"),
                        "category": parsed.get("category", "General Fiction"),
                        "series": parsed.get("series", "Standalone Novel"),
                        "protagonist": parsed.get("protagonist", "-"),
                        "evidence": parsed.get("evidence", "-"),
                        "cost_usd": 0.0,
                        "latency_ms": latency,
                        "status": "Success"
                    }
        except urllib.error.HTTPError as http_ex:
            err_msg = ""
            try:
                err_body = json.loads(http_ex.read().decode("utf-8"))
                err_msg = err_body.get("error", {}).get("message", str(http_ex))
            except Exception:
                err_msg = str(http_ex)
            last_error = f"HTTP {http_ex.code}: {err_msg}"
            # If 404 (model not found), try next model candidate
            if http_ex.code == 404:
                continue
            # For 400, 401, 403, stop and report immediately
            break
        except Exception as ex:
            last_error = f"{type(ex).__name__}: {ex}"
            break

    latency = round((time.time() - t0) * 1000, 1)
    return {
        "method": "Google AI Studio (Gemini)",
        "title": title,
        "author": author,
        "status": f"Error: {last_error}",
        "cost_usd": 0.0,
        "latency_ms": latency
    }
