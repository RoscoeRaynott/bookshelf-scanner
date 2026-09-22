import os
import re
import json
import time
import urllib.request
import urllib.parse

def run_vision_benchmark_openrouter(img_bgr, openrouter_key, model_id="google/gemini-3.8-flash"):
    """Benchmark bookshelf vision detection via OpenRouter."""
    import base64
    import cv2
    t0 = time.time()
    clean_key = str(openrouter_key).strip().strip("\"'")
    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return {"platform": "OpenRouter", "model": model_id, "status": "Failed to JPEG encode image", "latency_ms": 0, "cost_usd": 0.0, "books_count": 0, "books": []}
    b64_img = base64.b64encode(buffer).decode("utf-8")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {clean_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://bookshelf-scanner.streamlit.app",
        "X-Title": "Bookshelf Scanner Vision Arena",
    }
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this bookstore bookshelf image. Detect and catalog every book visible across all shelves from top to bottom, left to right.\nReturn strictly a JSON object:\n{\n  \"books\": [\n    {\n      \"shelf_row\": 1,\n      \"spine_text\": \"...\",\n      \"title\": \"...\",\n      \"author\": \"...\",\n      \"box_2d\": [ymin, xmin, ymax, xmax]\n    }\n  ]\n}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 16384,
        "reasoning": {"effort": "none"}
    }
    
    # Try with reasoning: effort none first, fallback if 400
    data = None
    last_err = ""
    for try_payload in [payload, {k: v for k, v in payload.items() if k != "reasoning"}]:
        req = urllib.request.Request(url, data=json.dumps(try_payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as h_err:
            if h_err.code == 400 and "reasoning" in try_payload:
                continue
            last_err = str(h_err)
            break
        except Exception as ex:
            last_err = str(ex)
            break

    latency_ms = round((time.time() - t0) * 1000, 1)
    if not data:
        return {
            "platform": "OpenRouter",
            "model": model_id,
            "status": f"Error: {last_err}",
            "latency_ms": latency_ms,
            "books_count": 0,
            "in_tokens": 0,
            "out_tokens": 0,
            "cost_usd": 0.0,
            "books": []
        }

    try:
        content = data["choices"][0]["message"]["content"]
        parsed = _parse_json_result(content)
        books = parsed.get("books", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 1185)
        completion_tokens = usage.get("completion_tokens", len(books) * 65)
        # OpenRouter Gemini 3.8 Flash rate: $0.75 / 1M in, $3.75 / 1M out
        cost = round((prompt_tokens / 1e6 * 0.75) + (completion_tokens / 1e6 * 3.75), 6)
        return {
            "platform": "OpenRouter",
            "model": model_id,
            "status": "Success",
            "latency_ms": latency_ms,
            "books_count": len(books),
            "in_tokens": prompt_tokens,
            "out_tokens": completion_tokens,
            "cost_usd": cost,
            "books": books
        }
    except Exception as ex:
        return {
            "platform": "OpenRouter",
            "model": model_id,
            "status": f"Error parsing: {ex}",
            "latency_ms": latency_ms,
            "books_count": 0,
            "in_tokens": 0,
            "out_tokens": 0,
            "cost_usd": 0.0,
            "books": []
        }


def run_vision_benchmark_direct_gemini(img_bgr, gemini_key, model_id="gemini-3.8-flash"):
    """Benchmark bookshelf vision detection directly via Google AI Studio API."""
    import base64
    import cv2
    t0 = time.time()
    clean_key = str(gemini_key).strip().strip("\"'")
    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return {"platform": "Direct Google AI Studio", "model": model_id, "status": "Failed to JPEG encode image", "latency_ms": 0, "cost_usd": 0.0, "books_count": 0, "books": []}
    b64_img = base64.b64encode(buffer).decode("utf-8")
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": clean_key
    }

    gen_configs = [
        {"responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}},
        {"responseMimeType": "application/json"}
    ]

    candidate_models = [model_id]
    for alt in ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]:
        if alt not in candidate_models:
            candidate_models.append(alt)

    raw_body = None
    last_err = ""
    used_model = model_id
    for m in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        for g_cfg in gen_configs:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Analyze this bookstore bookshelf image. Detect and catalog every book visible across all shelves from top to bottom, left to right.\nReturn strictly a JSON object:\n{\n  \"books\": [\n    {\n      \"shelf_row\": 1,\n      \"spine_text\": \"...\",\n      \"title\": \"...\",\n      \"author\": \"...\",\n      \"box_2d\": [ymin, xmin, ymax, xmax]\n    }\n  ]\n}"},
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": b64_img
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": g_cfg
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            raw_body, err = _execute_with_rate_limit_retry(req, max_retries=2, timeout=90)
            if raw_body:
                used_model = m
                break
            last_err = err or "Empty response"
            if "400" in str(last_err):
                # If thinkingConfig was not supported, retry with plain responseMimeType
                continue
            break
        if raw_body:
            break
        if "404" not in str(last_err):
            # If error is not a missing model ID, do not keep rotating models
            break

    latency_ms = round((time.time() - t0) * 1000, 1)
    if not raw_body:
        return {
            "platform": "Direct Google AI Studio",
            "model": used_model,
            "status": f"Error: {last_err}",
            "latency_ms": latency_ms,
            "books_count": 0,
            "in_tokens": 0,
            "out_tokens": 0,
            "cost_usd": 0.0,
            "books": []
        }
    text = _extract_text_from_resp(raw_body)
    parsed = _parse_json_result(text) if text else {}
    books = parsed.get("books", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
    
    usage_meta = raw_body.get("usageMetadata", {})
    prompt_tokens = usage_meta.get("promptTokenCount", 2200)
    completion_tokens = usage_meta.get("candidatesTokenCount", len(books) * 45)
    # Tier 1 Pay-As-You-Go rate: $0.075 / 1M in, $0.30 / 1M out
    cost = round((prompt_tokens / 1e6 * 0.075) + (completion_tokens / 1e6 * 0.30), 6)
    return {
        "platform": "Direct Google AI Studio",
        "model": used_model,
        "status": "Success",
        "latency_ms": latency_ms,
        "books_count": len(books),
        "in_tokens": prompt_tokens,
        "out_tokens": completion_tokens,
        "cost_usd": cost,
        "books": books
    }


def _extract_text_from_resp(raw_body):
    """Extract text from Interactions API or generateContent API response."""
    if "output_text" in raw_body and raw_body["output_text"]:
        return raw_body["output_text"]
    if "interaction" in raw_body and isinstance(raw_body["interaction"], dict):
        if "output_text" in raw_body["interaction"]:
            return raw_body["interaction"]["output_text"]
    candidates = raw_body.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"]
    if "text" in raw_body:
        return raw_body["text"]
    return None


def _parse_json_result(text):
    """Safely parse JSON response from LLM output, with fallback repairs for unescaped quotes and formatting anomalies."""
    if not text:
        return {"books": []}
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    t = t.strip()

    # Pass 1: standard json.loads
    try:
        return json.loads(t)
    except Exception:
        pass

    # Pass 2: candidate root object
    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last > first:
        candidate = t[first:last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

        # Fix trailing commas
        clean_commas = re.sub(r',\s*([\]}])', r'\1', candidate)
        try:
            return json.loads(clean_commas)
        except Exception:
            pass

        # Fix missing commas between objects
        fixed_objs = re.sub(r'\}\s*\{', '}, {', clean_commas)
        try:
            return json.loads(fixed_objs)
        except Exception:
            pass

    # Pass 3: salvage array if truncated
    if first != -1 and '"books"' in t:
        last_obj = t.rfind("}")
        if last_obj > first:
            repaired = t[first:last_obj + 1] + "\n  ]\n}"
            try:
                data = json.loads(repaired)
                if isinstance(data, dict) and "books" in data:
                    return data
            except Exception:
                pass

    # Pass 4: Regex object-by-object extraction (handles unescaped quotes inside string values)
    obj_matches = re.findall(r'\{[^{}]*(?:shelf_row|spine_text|title)[^{}]*\}', t)
    salvaged_books = []
    for obj_str in obj_matches:
        try:
            b_item = json.loads(obj_str)
            if isinstance(b_item, dict) and ("title" in b_item or "spine_text" in b_item):
                salvaged_books.append(b_item)
                continue
        except Exception:
            pass

        # Regex fallback for this single corrupted object
        s_row = re.search(r'["\']shelf_row["\']\s*:\s*(\d+)', obj_str)
        s_text = re.search(r'["\']spine_text["\']\s*:\s*["\'](.*?)["\']\s*(?:,|}|\n)', obj_str)
        t_text = re.search(r'["\']title["\']\s*:\s*["\'](.*?)["\']\s*(?:,|}|\n)', obj_str)
        a_text = re.search(r'["\']author["\']\s*:\s*["\'](.*?)["\']\s*(?:,|}|\n)', obj_str)
        box_match = re.search(r'["\']box_2d["\']\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', obj_str)
        box_val = [int(box_match.group(i)) for i in range(1, 5)] if box_match else [0, 0, 0, 0]
        if t_text or s_text:
            salvaged_books.append({
                "shelf_row": int(s_row.group(1)) if s_row else 1,
                "spine_text": s_text.group(1) if s_text else "-",
                "title": t_text.group(1) if t_text else "Book",
                "author": a_text.group(1) if a_text else "Unknown",
                "box_2d": box_val
            })

    if salvaged_books:
        return {"books": salvaged_books}

    raise ValueError(f"Could not parse or salvage JSON output: {t[:120]}")


def _execute_with_rate_limit_retry(req, max_retries=4, timeout=90):
    """Execute urllib request with automatic backoff on HTTP 429 quota/rate limits."""
    last_err = ""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), None
        except urllib.error.HTTPError as http_ex:
            err_msg = ""
            try:
                err_body = json.loads(http_ex.read().decode("utf-8"))
                err_msg = err_body.get("error", {}).get("message", str(http_ex))
            except Exception:
                err_msg = str(http_ex)
            last_err = f"HTTP {http_ex.code}: {err_msg}"
            
            # Rate limit backoff (HTTP 429)
            if http_ex.code == 429 and attempt < max_retries - 1:
                match = re.search(r"retry in ([\d\.]+)s", err_msg, re.IGNORECASE)
                wait_sec = float(match.group(1)) + 0.5 if match else (3.0 * (attempt + 1))
                time.sleep(wait_sec)
                continue
            
            return None, last_err
        except Exception as ex:
            return None, f"{type(ex).__name__}: {ex}"
    return None, last_err or "Exceeded max retries"


def query_direct_gemini_author_fame(author, gemini_api_key, preferred_model="gemini-3.5-flash-lite"):
    """Query Google AI Studio for author lifetime career sales ($0.00 / 4,000 RPM)."""
    if not gemini_api_key or not author:
        return {
            "author": author,
            "author_fame": "Not publicly reported",
            "author_fame_score": 0,
            "evidence": "-",
            "status": "Error: GEMINI_API_KEY or author is empty",
            "cost_usd": 0.0,
            "latency_ms": 0.0
        }

    clean_key = str(gemini_api_key).strip().strip("\"'")
    t0 = time.time()

    prompt = f"""You are an objective book industry research agent.
Analyze the author '{author}'.
Extract and return strictly a valid JSON object with:
- "author_fame": Author's verified total lifetime career book sales worldwide across all their works and formats (e.g., 'Over 100 million copies sold worldwide', 'Over 400 million books sold', '50 million copies sold', or 'Emerging / Midlist Author'). If unknown, write 'Not publicly reported'.
- "author_fame_score": Integer total lifetime copies sold (e.g. 100000000 for 100M, 50000000 for 50M, 0 if unknown).
- "evidence": 1-sentence source summary.

Output ONLY the JSON object, no commentary."""

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": clean_key
    }
    last_error = ""

    candidate_models = [preferred_model]
    for m in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]:
        if m not in candidate_models:
            candidate_models.append(m)

    gen_configs = [
        {"responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "minimal"}},
        {"responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}},
        {"responseMimeType": "application/json"}
    ]

    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for g_cfg in gen_configs:
            payload_gc = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": g_cfg
            }
            req = urllib.request.Request(url, data=json.dumps(payload_gc).encode("utf-8"), headers=headers)
            raw_body, err = _execute_with_rate_limit_retry(req, max_retries=3)
            if raw_body:
                extracted_text = _extract_text_from_resp(raw_body)
                if extracted_text:
                    parsed = _parse_json_result(extracted_text)
                    latency = round((time.time() - t0) * 1000, 1)
                    return {
                        "author": author,
                        "author_fame": parsed.get("author_fame", "Not publicly reported"),
                        "author_fame_score": int(parsed.get("author_fame_score", 0) or 0),
                        "evidence": parsed.get("evidence", "-"),
                        "cost_usd": 0.0,
                        "latency_ms": latency,
                        "status": "Success"
                    }
            else:
                last_error = err or "Empty response"
                if "HTTP 400" in last_error:
                    continue
                if "HTTP 404" in last_error:
                    break
                break

    latency = round((time.time() - t0) * 1000, 1)
    return {
        "author": author,
        "author_fame": "Not publicly reported",
        "author_fame_score": 0,
        "evidence": "-",
        "status": f"Error: {last_error}",
        "cost_usd": 0.0,
        "latency_ms": latency
    }


def query_direct_gemini_api(title, author, gemini_api_key, preferred_model="gemini-3.5-flash-lite"):
    """Query Google AI Studio Gemini API ($0.00 on Free Tier / ~$0.0001 on Pay-As-You-Go)."""
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

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": clean_key
    }
    last_error = ""

    # Priority models list starting with preferred_model
    candidate_models = [preferred_model]
    for m in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]:
        if m not in candidate_models:
            candidate_models.append(m)
    
    # Try minimal thinking first to minimize latency
    gen_configs = [
        {"responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "minimal"}},
        {"responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}},
        {"responseMimeType": "application/json"}
    ]

    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for g_cfg in gen_configs:
            payload_gc = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": g_cfg
            }
            req = urllib.request.Request(url, data=json.dumps(payload_gc).encode("utf-8"), headers=headers)
            raw_body, err = _execute_with_rate_limit_retry(req, max_retries=3)
            if raw_body:
                extracted_text = _extract_text_from_resp(raw_body)
                if extracted_text:
                    parsed = _parse_json_result(extracted_text)
                    latency = round((time.time() - t0) * 1000, 1)
                    cfg_label = "minimal-thinking" if "thinkingConfig" in g_cfg else "default"
                    return {
                        "method": f"Google AI Studio ({model} • {cfg_label})",
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
            else:
                last_error = err or "Empty response"
                # If 400 Bad Request (e.g. unsupported thinking config parameter), try next config
                if "HTTP 400" in last_error:
                    continue
                # If 404 Not Found (model does not exist), break to try next model candidate
                if "HTTP 404" in last_error:
                    break
                break

    latency = round((time.time() - t0) * 1000, 1)
    return {
        "method": "Google AI Studio",
        "title": title,
        "author": author,
        "status": f"Error: {last_error}",
        "cost_usd": 0.0,
        "latency_ms": latency
    }


def draw_annotated_vision_result(img_bgr, books):
    """Draw translucent shelf-colored bounding boxes and ID badges on shelf image for side-by-side comparison."""
    import cv2
    import numpy as np
    from PIL import Image

    if img_bgr is None:
        return None
    h_img, w_img = img_bgr.shape[:2]
    scale_factor = max(1.0, h_img / 1200.0)
    box_thickness = max(2, int(2.0 * scale_factor))
    font_scale = 0.38 * scale_factor
    font_thick = max(1, int(1.2 * scale_factor))

    annotated = img_bgr.copy()
    overlay = img_bgr.copy()
    colors = [(50, 220, 100), (240, 150, 40), (220, 60, 220), (30, 200, 240), (255, 100, 50), (100, 100, 255)]

    valid_books = []
    for idx, b in enumerate(books, start=1):
        box = b.get("box_2d", [0, 0, 0, 0])
        if len(box) == 4 and any(v > 0 for v in box):
            ymin, xmin, ymax, xmax = box
            py_min = max(0, min(h_img - 1, int((ymin / 1000.0) * h_img)))
            px_min = max(0, min(w_img - 1, int((xmin / 1000.0) * w_img)))
            py_max = max(0, min(h_img - 1, int((ymax / 1000.0) * h_img)))
            px_max = max(0, min(w_img - 1, int((xmax / 1000.0) * w_img)))
            if px_max > px_min and py_max > py_min:
                shelf = int(b.get("shelf_row", 1) or 1)
                valid_books.append((idx, shelf, px_min, py_min, px_max, py_max))

    # 1. Translucent fill
    for idx, shelf, px_min, py_min, px_max, py_max in valid_books:
        col = colors[(shelf - 1) % len(colors)]
        cv2.rectangle(overlay, (px_min, py_min), (px_max, py_max), col, -1)

    cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)

    # 2. Crisp borders & dynamic ID badges
    for idx, shelf, px_min, py_min, px_max, py_max in valid_books:
        col = colors[(shelf - 1) % len(colors)]
        cv2.rectangle(annotated, (px_min, py_min), (px_max, py_max), col, box_thickness)
        
        text = str(idx)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        pad_x = int(6 * scale_factor)
        pad_y = int(4 * scale_factor)
        badge_w = tw + pad_x * 2
        badge_h = th + pad_y * 2
        
        badge_x1 = max(0, min(w_img - badge_w - 1, px_min))
        badge_y1 = max(0, py_min - badge_h) if py_min >= badge_h else py_min
        badge_x2 = min(w_img - 1, badge_x1 + badge_w)
        badge_y2 = min(h_img - 1, badge_y1 + badge_h)
        
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (15, 15, 15), -1)
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), col, max(1, int(1.0 * scale_factor)))
        
        cv2.putText(annotated, text, (badge_x1 + pad_x, badge_y1 + th + pad_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
