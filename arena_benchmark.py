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
        "max_tokens": 4096,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            latency_ms = round((time.time() - t0) * 1000, 1)
            content = data["choices"][0]["message"]["content"]
            parsed = _parse_json_result(content)
            books = parsed.get("books", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 2200)
            completion_tokens = usage.get("completion_tokens", len(books) * 45)
            # OpenRouter Gemini 3.8 Flash rate: $0.75 / 1M in, $3.75 / 1M out
            cost = round((prompt_tokens / 1e6 * 0.75) + (completion_tokens / 1e6 * 3.75), 5)
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
        latency_ms = round((time.time() - t0) * 1000, 1)
        return {
            "platform": "OpenRouter",
            "model": model_id,
            "status": f"Error: {ex}",
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
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    candidate_models = [model_id]
    for alt in ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]:
        if alt not in candidate_models:
            candidate_models.append(alt)

    raw_body = None
    last_err = ""
    used_model = model_id
    for m in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        raw_body, err = _execute_with_rate_limit_retry(req, max_retries=2)
        if raw_body:
            used_model = m
            break
        last_err = err or "Empty response"
        if "404" not in str(last_err):
            # If error is quota or network, don't keep cycling models
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
    """Safely parse JSON response from LLM output."""
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        match = re.search(r'\{[\s\S]*\}', t)
        if match:
            return json.loads(match.group(0))
        raise

def _execute_with_rate_limit_retry(req, max_retries=4):
    """Execute urllib request with automatic backoff on HTTP 429 quota/rate limits."""
    last_err = ""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
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
