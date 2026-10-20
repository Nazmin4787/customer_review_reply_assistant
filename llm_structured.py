"""Step A: one structured LLM call per review (JSON out), with validation,
rate-limit-aware retries, caching, and per-call latency/token logging.

Usage in the notebook:
    from llm_structured import run_batch
    results = run_batch(sample_df, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

PROMPT_VERSION = "v4-json"

CATEGORIES = [
    "product_performance",
    "defect_or_failure",
    "delivery_or_service",
    "refund_or_resolution",
    "negative_experience",
    "urgent_or_safety",
    "other",
]
URGENCY_LEVELS = ["low", "medium", "high"]
REQUIRED_KEYS = {"category", "summary", "response", "urgency"}

SYSTEM_PROMPT = f"""You are a Senior Customer Support Specialist at a large online retailer.
You read ONE customer review and return ONLY a JSON object, no extra text, with exactly these keys:

"category": one of {CATEGORIES}. Pick the customer's MAIN complaint.
"summary": one neutral sentence (max 25 words) describing the main problem.
"urgency": one of ['low', 'medium', 'high'].
   high   = safety hazard (fire, shock, injury); OR the product is completely unusable or dead;
            OR the customer has contacted support repeatedly without resolution and is
            threatening escalation (consumer complaint, legal action, chargeback).
   medium = a clear defect or a single unresolved support/warranty problem, but the product
            still works at least partly.
   low    = quality, price, expectation or minor dissatisfaction; no functional failure.
"response": a draft reply email as plain text (90 to 135 words) with this structure:
   Subject: <calm subject naming the product and main problem>
   (blank line)
   Dear <Full Name if the customer's own name appears in the review, else Valued Customer>,
   (blank line)
   Paragraph 1 (2 sentences): sincere apology for the top one or two problems, showing understanding.
   Paragraph 2 (2 sentences): ask the customer to reply with their order number or invoice. If the
   customer asked for a refund/replacement/repair say the support team "can review whether this
   request is eligible"; otherwise say the team "can review the available options".
   (blank line)
   Best regards,
   Customer Support Team

RESPONSE RULES
- Use "we" / "our support team", never "I". Plain, warm English. Sentences under 25 words.
- Mention at most TWO problems. Only use facts written in the review.
- Never invent or promise order numbers, dates, amounts, policies, warranties, compensation or actions.
- Never write: escalated, approved, processed, investigated, resolved, refunded, replaced, "we will".
- Do not blame or comment on third parties; do not admit liability; avoid "unacceptable".
- Do not mention the rating, these rules, or this JSON format inside the response.
- No filler or soft promises: never use "as soon as possible", "in a timely manner", "quickly", "satisfactory solution"

URGENCY CALIBRATION EXAMPLES (invented, for calibration only)
- "Charger sparked and there was a burning smell" -> high
- "Phone dead after an update; three calls to support, no fix; will file a consumer complaint" -> high
- "Phone works but the camera is average" -> low
- "Speaker crackles at high volume; seller replied but has not sent a replacement yet" -> medium
"""

FORBIDDEN_PHRASES = [
    "escalated", "we have investigated", "investigat", "we will", "unacceptable",
    "our fault", "we failed", "refund has been", "replacement has been",
    "has been approved", "has been processed", "we have resolved", "guaranteed",
    "as soon as possible", "in a timely manner", "satisfactory solution"
]


# ---------------------------------------------------------------- validation
def parse_json_reply(text: str) -> dict:
    """Parse the model output, tolerating ```json fences or stray text around the object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_output(data: dict) -> list[str]:
    """Return a list of problems; empty list means the output is acceptable."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["output is not a JSON object"]

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        return [f"missing keys: {sorted(missing)}"]

    if data["category"] not in CATEGORIES:
        problems.append(f"invalid category: {data['category']!r}")
    if str(data["urgency"]).lower() not in URGENCY_LEVELS:
        problems.append(f"invalid urgency: {data['urgency']!r}")

    summary = str(data["summary"]).strip()
    if not summary or len(summary.split()) > 35:
        problems.append("summary empty or longer than 35 words")

    resp = str(data["response"]).strip()
    low = resp.lower()
    words = len(resp.split())
    if not resp.startswith("Subject:"):
        problems.append("response must start with 'Subject:'")
    if not re.search(r"(?im)^dear\s+[^,\n]{2,80},\s*$", resp):
        problems.append("missing 'Dear ...,' greeting")
    if "best regards," not in low or "customer support team" not in low:
        problems.append("missing closing/signature")
    if not 70 <= words <= 150:
        problems.append(f"response length {words} words (want 90-135)")
    if re.search(r"\bI['’]?m\b|\bI\s+(understand|am|apologi[sz]e|will)\b", resp):
        problems.append("uses first person singular")
    for phrase in FORBIDDEN_PHRASES:
        if phrase in low:
            problems.append(f"risky wording: '{phrase}'")
    return problems


def baseline_urgency(severity_score: float) -> str:
    """Rule-based urgency from your severity score, for comparison with the LLM's urgency.
    Thresholds are a starting point: calibrate them on your own score distribution."""
    if severity_score >= 200:
        return "high"
    if severity_score >= 120:
        return "medium"
    return "low"


# ---------------------------------------------------------------- API call
def _retry_delay(error_text: str, attempt: int) -> float:
    """Use the wait time Groq/OpenAI suggest ('try again in 9.78s'); else exponential backoff."""
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)", error_text)
    if m:
        value = float(m.group(1))
        return (value / 1000 if m.group(2) == "ms" else value) + 1.0
    return min(2 ** attempt * 3, 60)


def _is_fatal_quota(status: int, text: str) -> bool:
    """Only genuinely exhausted quota/credits are fatal. Per-minute 429s are NOT."""
    t = text.lower()
    return (
        status == 402
        or "insufficient_quota" in t
        or "per day" in t
        or "tokens per day" in t
        or "exceeded your current quota" in t
    )


def _post(api_key, base_url, model, messages, json_mode, timeout=60):
    payload = {"model": model, "messages": messages, "temperature": 0.3}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return requests.post(
        f"{(base_url or 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )


def analyze_review(review_text: str, rating, api_key: str, base_url: str | None,
                   model: str, max_attempts: int = 5) -> dict:
    """One structured call (+ one repair call if validation fails).
    Always returns a dict with a 'status' field: ok | invalid | quota | error."""
    user_msg = (
        f"Star rating: {rating} of 5\n"
        f"--- REVIEW START ---\n{str(review_text).strip()[:4000]}\n--- REVIEW END ---"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}]

    out = {"status": "error", "error": None, "latency_s": 0.0,
           "prompt_tokens": 0, "completion_tokens": 0, "attempts": 0}
    json_mode = True
    repaired = False
    start = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        out["attempts"] = attempt
        try:
            resp = _post(api_key, base_url, model, messages, json_mode)
        except requests.RequestException as exc:
            out["error"] = f"network: {exc}"
            time.sleep(_retry_delay("", attempt))
            continue

        if resp.status_code >= 400:
            text = resp.text[:600]
            out["error"] = f"{resp.status_code}: {text}"
            if _is_fatal_quota(resp.status_code, text):
                out["status"] = "quota"
                break
            if resp.status_code == 400 and json_mode:     # model may not support json_object
                json_mode = False
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(_retry_delay(text, attempt))
                continue
            break                                         # other 4xx: retrying won't help

        body = resp.json()
        usage = body.get("usage", {})
        out["prompt_tokens"] += usage.get("prompt_tokens", 0)
        out["completion_tokens"] += usage.get("completion_tokens", 0)
        content = (body["choices"][0]["message"].get("content") or "").strip()

        try:
            data = parse_json_reply(content)
            problems = validate_output(data)
        except json.JSONDecodeError as exc:
            data, problems = None, [f"invalid JSON: {exc}"]

        if not problems:
            data["urgency"] = str(data["urgency"]).lower()
            out.update(data)
            out.update(status="ok", error=None)
            break

        out.update(status="invalid", error="; ".join(problems), raw=content)
        if repaired:
            break
        repaired = True                                   # one repair attempt, with feedback
        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": "Fix these problems and return ONLY the corrected JSON: "
                                         + "; ".join(problems)},
        ]

    out["latency_s"] = round(time.perf_counter() - start, 2)
    return out


# ---------------------------------------------------------------- batch + cache
def run_batch(df: pd.DataFrame, api_key: str, base_url: str | None, model: str,
              text_col: str = "customer_message", rating_col: str = "Rating",
              id_col: str = "Unique_ID", severity_col: str | None = "severity_score",
              cache_path: str | Path = "outputs/structured_cache.json",
              min_interval_s: float = 2.0, verbose: bool = True) -> pd.DataFrame:
    """Analyze every row of df. Cached 'ok' results are reused, so re-runs cost nothing.
    Stops early (keeping what it has) if the daily quota is exhausted."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    key_prefix = f"{PROMPT_VERSION}|{model}|"

    rows, last_call = [], 0.0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        case_id = str(row[id_col])
        key = key_prefix + case_id
        cached = cache.get(key)

        if cached and cached.get("status") == "ok":
            result, source = cached, "cache"
        else:
            wait = min_interval_s - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            result = analyze_review(row[text_col], row[rating_col], api_key, base_url, model)
            last_call = time.time()
            source = "api"
            if result["status"] != "error":
                cache[key] = result
                cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

        record = {"case_id": case_id, "rating": row[rating_col], "source": source, **result}
        if severity_col and severity_col in df.columns:
            record["severity_score"] = row[severity_col]
            record["baseline_urgency"] = baseline_urgency(row[severity_col])
        rows.append(record)

        if verbose:
            print(f"[{i}/{len(df)}] case {case_id}: {result['status']} ({source}, {result['latency_s']}s)")
        if result["status"] == "quota":
            print("Daily quota exhausted: stopping. Re-run later; cached results are kept.")
            break

    return pd.DataFrame(rows)
