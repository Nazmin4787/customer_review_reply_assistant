"""Reusable pipeline for the Streamlit app: clean -> filter 1-2 star -> LLM -> safe fallback.

Depends on llm_structured.py (prompt, API call, validation) and eval_tools.py (grounding check).
"""
from __future__ import annotations

import hashlib
import re
import time

import pandas as pd
import requests

from eval_tools import grounding_flags
from llm_structured import CATEGORIES, PROMPT_VERSION, URGENCY_LEVELS, analyze_review, parse_json_reply

# USD per 1M tokens (gpt-oss-120b on Groq). Confirm on the provider's pricing page.
PRICE_IN_PER_1M = 0.15
PRICE_OUT_PER_1M = 0.60

URGENCY_ORDER = {"high": 0, "medium": 1, "low": 2}

FALLBACK_REPLY = (
    "Subject: Your recent review\n\n"
    "Dear Valued Customer,\n\n"
    "We are sorry to hear about your experience with this product. We understand how "
    "disappointing this is, and we appreciate you taking the time to tell us.\n\n"
    "Please reply with your order number or invoice so our support team can review the "
    "available options.\n\n"
    "Best regards,\nCustomer Support Team"
)


# Wording that promises an action, a timeline or an outcome. The prompt forbids it, but the model
# still produces it, so such replies are flagged for a person instead of being trusted.
_SOFT_PROMISES = [
    (r"\bkeeps? you (updated|informed|posted)\b", "keeps you updated/informed"),
    (r"\b(updated|informed) regularly\b", "updated/informed regularly"),
    (r"\blet you know\b", "let you know"),
    (r"\b(soon|shortly|right away|immediately)\b", "promises speed"),
    (r"\b(quickly|promptly)\s+(review|resolve|address|locate|look|investigate|assist|respond|help|get|find|"
     r"check|examine|fix|sort|handle|follow)\b|\b(review|resolve|address|locate|assist|respond|help|find|check|"
     r"examine|fix|handle|follow up)\b[^.]{0,40}\b(quickly|promptly)\b", "promises speed"),
    (r"\bwe(\'ll| will)\b", "we will"),
    (r"\blook(ing)? into\b", "look into"),
    (r"\b(find|provide|reach) (you )?a (suitable |satisfactory |quick )?(solution|resolution)\b", "promises a solution"),
    (r"\b(ensure|guarantee)\b", "ensure/guarantee"),
    (r"\binvestigat", "investigate"),
    (r"\b(resolve|resolving)\b", "resolve"),
    (r"\bsatisfactor", "satisfactory"),
    (r"\b(thorough|thoroughly) (examin|review|investigat)", "thoroughly examine"),
    (r"\bdetermine the (appropriate|best|right|most suitable)\b", "determine next steps"),
]


def soft_promises(reply: str, review: str = "") -> list[str]:
    """Promise-like wording in the reply. Words the customer used themselves are ignored."""
    low = str(reply).lower().replace("\u2019", "'")
    src = str(review).lower().replace("\u2019", "'")
    found = []
    for pat, label in _SOFT_PROMISES:
        hits = [m.group(0) for m in re.finditer(pat, low)]
        if hits and not all(h in src for h in hits):
            found.append(label)
    return found


def check_reply(review: str, reply: str) -> list[str]:
    """Reasons a person should look at this reply (empty list = nothing automatic found)."""
    reasons = []
    flags = grounding_flags(review, reply)
    if flags:
        reasons.append("mentions remedy not in review: " + ", ".join(flags))
    promises = soft_promises(reply, review)
    if promises:
        reasons.append("soft promise: " + ", ".join(promises))
    return reasons


# ------------------------------------------------------------------ rebuild the reply safely
# The model is not reliable at *not* writing promises, so the app does not trust its wording:
#   - the "send your order number" part is a fixed sentence, never model text;
#   - model sentences with promise-like wording or a remedy the customer never asked for are removed;
#   - invented context ("in your home", "today") is trimmed out.
# Everything removed is reported, so a person can see what changed.
_REQUEST_RE = re.compile(r"order number|invoice|purchase confirmation|proof of purchase|order id|"
                         r"purchase details|reply to this email|that information|this information|"
                         r"once we have|once you (?:send|provide|share)|upon receiv", re.I)
_REMEDY_IN_REVIEW = re.compile(r"refund|money back|chargeback|replac|exchange|repair|return|pick ?up|"
                               r"warrant|guarantee", re.I)
_UNSUPPORTED_PHRASES = [
    r"\s+(?:in|at) your (?:home|house|office|kitchen|room)\b",
    r"\s+(?:today|yesterday|tomorrow|tonight|recently)\b",
    r"\s+(?:this|last) (?:week|month|morning|evening|night)\b",
]
GENERIC_APOLOGY = "We are sorry to hear about your experience with this product."
MAX_KEPT_SENTENCES = 3


def _split_sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def _strip_unsupported(review: str, sentence: str) -> tuple[str, list[str]]:
    low_review, trimmed = review.lower(), []

    def drop(match):
        if match.group(0).strip().lower() in low_review:
            return match.group(0)
        trimmed.append(match.group(0).strip())
        return ""

    for pat in _UNSUPPORTED_PHRASES:
        sentence = re.sub(pat, drop, sentence, flags=re.I)
    return sentence, trimmed


def sanitize_reply(review: str, reply: str, mismatch: bool = False) -> tuple[str, str]:
    """Return (safe_reply, note). `note` lists what was removed or trimmed ('' if nothing)."""
    lines = str(reply).replace("\r\n", "\n").strip().split("\n")
    stripped = [l.strip() for l in lines]
    subject = next((l for l in stripped if l.lower().startswith("subject:")), "Subject: Your recent review")
    if soft_promises(subject, review) or (mismatch and (_PRAISE_RE.search(subject) or "star" in subject.lower())):
        subject = "Subject: Your recent review"
    g_idx = next((i for i, l in enumerate(stripped) if re.match(r"(?i)^dear\s+[^,\n]{2,80},\s*$", l)), None)
    greeting = stripped[g_idx] if g_idx is not None else "Dear Valued Customer,"
    s_idx = next((i for i, l in enumerate(stripped) if l.lower().startswith("subject:")), -1)
    start = (g_idx if g_idx is not None else s_idx) + 1
    end = next((i for i, l in enumerate(stripped) if i >= start and l.lower().startswith("best regards")),
               len(stripped))
    body = " ".join(l for l in stripped[start:end] if l)

    kept, notes = [], []
    for sentence in _split_sentences(body):
        if _REQUEST_RE.search(sentence):
            continue                                   # replaced by the fixed request sentence below
        if mismatch and _PRAISE_RE.search(sentence):
            notes.append(f'removed "{sentence}" (praise wording on a very negative review)')
            continue
        sentence, trimmed = _strip_unsupported(review, sentence)
        why = []
        why += soft_promises(sentence, review)
        why += ["remedy not in review: " + ", ".join(grounding_flags(review, sentence))
                for _ in [0] if grounding_flags(review, sentence)]
        if why:
            notes.append(f'removed "{sentence}" ({"; ".join(why)})')
            continue
        if trimmed:
            notes.append("trimmed invented context: " + ", ".join(trimmed))
        kept.append(sentence)
    kept = kept[:MAX_KEPT_SENTENCES] or [GENERIC_APOLOGY]

    eligible = bool(_REMEDY_IN_REVIEW.search(review))
    request = ("Please reply with your order number or invoice so our support team can review whether "
               "this request is eligible." if eligible else
               "Please reply with your order number or invoice so our support team can review the "
               "available options.")
    if mismatch:
        request = f"{request} {MISMATCH_NOTE}"
    clean = (f"{subject}\n\n{greeting}\n\n{' '.join(kept)}\n\n{request}\n\n"
             "Best regards,\nCustomer Support Team")
    return clean, " | ".join(notes)


# ------------------------------------------------------------------ severity ranking
# Same idea as the notebook's explainable severity score (rating + detail + complaint words).
# Differences: simpler tokenising, and a few extra safety words (spark, smoke, burnt).
_STOP = {"the", "and", "was", "were", "this", "that", "with", "from", "have", "has", "had", "for", "are",
         "but", "they", "them", "you", "your", "our", "their", "just", "also", "one", "use", "used",
         "using", "amazon", "review", "reviews", "customer", "its", "very", "too", "can", "will", "get"}
_DEFECT = {"broken", "defective", "damaged", "stopped", "failed", "failure", "dead", "cracked", "leaking",
           "overheating", "malfunction", "faulty", "missing", "incomplete", "unusable"}
_SERVICE = {"delivery", "delivered", "delayed", "delay", "late", "courier", "packaging", "seller", "service",
            "support", "response", "communication", "installation", "technician", "rude"}
_RESOLUTION = {"refund", "replacement", "exchange", "return", "warranty", "compensation", "repair", "claim",
               "cashback", "moneyback", "reimburse"}
_EXPERIENCE = {"poor", "bad", "worst", "worse", "disappointing", "disappointed", "useless", "waste", "issue",
               "problem", "unhappy", "frustrated", "cheating", "fraud", "overpriced", "unacceptable",
               "avoid", "regret"}
_URGENCY = {"urgent", "immediately"}
_SAFETY_WORDS = {"unsafe", "dangerous", "fire", "explode", "explosion", "injury",
                 "spark", "sparks", "sparked", "sparking", "smoke", "smoking", "burnt"}
_SAFETY_PHRASES = ("safety issue", "safety concern", "safety risk", "health risk", "fire hazard", "caught fire",
                   "electrical fire", "electric shock", "got shocked", "received a shock", "caused injury",
                   "physical injury", "dangerous product", "dangerous situation", "risk of fire",
                   "risk of injury", "burning smell")


def _count(tokens: list[str], vocab: set, cap: int) -> int:
    return min(sum(t in vocab for t in tokens), cap)


def severity_score(review: str, rating) -> int:
    """Higher = more urgent. Rating, detail (capped at 100), then capped counts of complaint words."""
    low = str(review).lower().replace("\u2019", "'")
    tokens = [t for t in re.findall(r"[a-z]+", low) if len(t) > 2 and t not in _STOP]
    score = 30 if float(rating) <= 1 else 15
    score += min(len(tokens), 100)
    score += 20 * _count(tokens, _URGENCY, 3)
    if any(p in low for p in _SAFETY_PHRASES) or any(t in _SAFETY_WORDS for t in tokens):
        score += 30
    score += 10 * _count(tokens, _DEFECT, 5)
    score += 8 * _count(tokens, _RESOLUTION, 3)
    score += 5 * _count(tokens, _EXPERIENCE | _SERVICE, 5)
    return int(score)


def rank_reviews(prepared: pd.DataFrame) -> pd.DataFrame:
    """Add file_pos, word_count, severity_score and rank (1 = most critical)."""
    out = prepared.copy().reset_index(drop=True)
    out["file_pos"] = range(len(out))
    out["word_count"] = out["review"].str.split().map(len)
    out["severity_score"] = [severity_score(r, g) for r, g in zip(out["review"], out["rating"])]
    order = out.sort_values(["severity_score", "word_count", "file_pos"],
                            ascending=[False, False, True], kind="stable").index
    out["rank"] = 0
    out.loc[order, "rank"] = range(1, len(out) + 1)
    return out


def select_reviews(ranked: pd.DataFrame, n: int, by: str = "critical") -> pd.DataFrame:
    """First n reviews: by='critical' = highest severity first, by='file' = order in the file."""
    key = "rank" if by == "critical" else "file_pos"
    return ranked.sort_values(key, kind="stable").head(int(n)).reset_index(drop=True)



# ------------------------------------------------------------------ text sentiment (rule-based, no API call)
_NEG_STRONG = {"worst", "terrible", "horrible", "awful", "useless", "scam", "fraud", "garbage", "junk", "trash",
               "rubbish", "defective", "dangerous", "unsafe", "waste", "cheated", "cheating", "disgusting",
               "pathetic", "unacceptable", "disappointed", "disappointing", "regret", "faulty", "refund"}
_NEG_PHRASES = ("do not buy", "don't buy", "dont buy", "never buy", "never again", "stopped working",
                "not working", "doesn't work", "does not work", "money back", "waste of money", "burning smell",
                "fell apart", "broke after", "caught fire", "rip off", "ripoff", "so bad", "very bad", "too bad")
_NEG_WORDS = {"broken", "broke", "cracked", "burnt", "burned", "sparked", "sparks", "smoke", "leaking", "dead",
              "poor", "bad", "problem", "issue", "refuses", "refused", "rude", "late", "damaged", "missing",
              "fail", "failed", "failure", "frustrated", "unhappy", "angry", "complaint", "overheating"}
_POS_WORDS = {"love", "loved", "great", "perfect", "excellent", "amazing", "awesome", "fantastic", "happy",
              "satisfied", "recommend", "recommended", "wonderful", "superb", "good", "best", "worth", "nice",
              "pleased", "impressed", "comfortable"}
_POS_PHRASES = ("works perfectly", "works great", "works well", "highly recommend", "value for money")


def text_sentiment(review: str) -> tuple[str, int]:
    """Judge the text only (not the stars). Returns (label, net) with label in
    very_negative / negative / mixed / positive / neutral. net = negative hits - positive hits."""
    low = str(review).lower().replace("\u2019", "'")
    tokens = re.findall(r"[a-z']+", low)
    neg = 2 * sum(t in _NEG_STRONG for t in tokens) + sum(t in _NEG_WORDS for t in tokens)
    neg += 2 * sum(p in low for p in _NEG_PHRASES)
    pos = sum(t in _POS_WORDS for t in tokens) + 2 * sum(p in low for p in _POS_PHRASES)
    if neg >= 4 and neg >= 2 * pos:
        label = "very_negative"
    elif neg >= 2 and neg > pos:
        label = "negative"
    elif neg >= 2 and pos >= 2:
        label = "mixed"
    elif pos >= 2 and pos > neg:
        label = "positive"
    else:
        label = "neutral"
    return label, neg - pos


def is_mismatch(rating, review: str) -> bool:
    """High star rating (4-5) but a very negative text: a likely mis-click, sarcasm or a complaint."""
    try:
        high = float(rating) >= 4
    except (TypeError, ValueError):
        return False
    return high and text_sentiment(review)[0] == "very_negative"


_PRAISE_RE = re.compile(r"thank(s| you)[^.!?]{0,40}(rating|stars?|feedback|kind|positive|great|high)|"
                        r"glad (you|to hear)|happy (to hear|that you)|pleased (to hear|that you)|"
                        r"(5|five)[- ]star|enjoy(ing)? your", re.I)
MISMATCH_NOTE = "If your star rating does not reflect your experience, you are welcome to update it."

# ------------------------------------------------------------------ preparation
def guess_column(columns: list[str], patterns: list[str]) -> str | None:
    """First column whose lower-cased name contains one of the patterns (in pattern order)."""
    lowered = {c: str(c).lower() for c in columns}
    for pat in patterns:
        for col, low in lowered.items():
            if pat in low:
                return col
    return None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def prepare_reviews(raw: pd.DataFrame, text_col: str, rating_col: str,
                    header_col: str | None = None, id_col: str | None = None,
                    max_rating: int = 2, include_mismatch: bool = False) -> tuple[pd.DataFrame, dict]:
    """Return (critical reviews with columns case_id, rating, review) and a dict of counts."""
    df = raw.copy()
    text = df[text_col].fillna("").astype(str).map(_clean)
    if header_col:
        header = df[header_col].fillna("").astype(str).map(_clean)
        review = [f"{h.rstrip('.')}. {t}" if h and t else (h or t) for h, t in zip(header, text)]
    else:
        review = text.tolist()

    prepared = pd.DataFrame({
        "case_id": df[id_col].astype(str).values if id_col else (df.index + 1).astype(str),
        "rating": pd.to_numeric(df[rating_col], errors="coerce").values,
        "review": review,
    })
    stats = {"rows_in": len(prepared)}

    bad_rating = prepared["rating"].isna()
    stats["dropped_bad_rating"] = int(bad_rating.sum())
    prepared = prepared[~bad_rating]

    empty = prepared["review"].str.len() == 0
    stats["dropped_empty_text"] = int(empty.sum())
    prepared = prepared[~empty]

    before = len(prepared)
    prepared = prepared.drop_duplicates(subset=["review", "rating"])
    stats["dropped_duplicates"] = before - len(prepared)

    critical = prepared["rating"] <= max_rating
    mismatch = pd.Series([is_mismatch(g, r) for g, r in zip(prepared["rating"], prepared["review"])],
                         index=prepared.index)
    stats["mismatch_available"] = int(mismatch.sum())
    keep = critical | mismatch if include_mismatch else critical
    stats["not_critical"] = int((~keep).sum())
    prepared = prepared[keep].copy()
    prepared["mismatch"] = mismatch[keep].values
    prepared = prepared.reset_index(drop=True)
    stats["critical"] = int((prepared["rating"] <= max_rating).sum())
    stats["mismatch"] = int(prepared["mismatch"].sum())
    return prepared, stats


# ------------------------------------------------------------------ one review
def _cost(prompt_tokens: float, completion_tokens: float, prices=None) -> float:
    price_in, price_out = prices or (PRICE_IN_PER_1M, PRICE_OUT_PER_1M)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def _salvage(raw: str) -> dict | None:
    """Usable fields from a reply the validator rejected, or None if the JSON itself is unusable."""
    try:
        data = parse_json_reply(raw)
        category, urgency = data["category"], str(data["urgency"]).lower()
        summary, response = str(data["summary"]).strip(), str(data["response"]).strip()
    except Exception:
        return None
    if category not in CATEGORIES or urgency not in URGENCY_LEVELS or not summary or not response:
        return None
    return {"category": category, "urgency": urgency, "summary": summary, "response": response}


def process_review(case_id: str, rating, review: str, api_key: str, base_url: str, model: str,
                   prices=None) -> dict:
    """LLM analysis with a safe fallback. Never raises; always returns a full record."""
    result = analyze_review(review, rating, api_key, base_url, model)
    llm_status = result.get("status", "error")
    validator_note = ""
    if llm_status == "invalid" and result.get("raw"):
        salvaged = _salvage(result["raw"])
        if salvaged:                      # valid JSON, only the validator's wording/length rules failed
            validator_note = f"validator complaint ignored because the reply is rebuilt: {str(result.get('error'))[:120]}"
            result = {**result, **salvaged}
            llm_status = "ok"
    rec = {
        "case_id": case_id, "rating": rating, "review": review, "model": model,
        "llm_status": llm_status,
        "latency_s": result.get("latency_s", 0.0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "error": result.get("error"),
    }
    rec["cost_usd"] = _cost(rec["prompt_tokens"], rec["completion_tokens"], prices)

    reasons = []
    sentiment = text_sentiment(review)[0]
    mismatch = is_mismatch(rating, review)
    rec["sentiment"], rec["mismatch"] = sentiment, mismatch
    if mismatch:
        reasons.append(f"rating/text mismatch: {rating:g} stars but very negative text; check for sarcasm "
                       "or a mis-click")
    if llm_status == "ok":
        clean, note = sanitize_reply(review, result["response"], mismatch)
        rec.update(status="ok", category=result["category"], urgency=result["urgency"],
                   summary=result["summary"], reply=clean, model_reply=result["response"],
                   auto_cleaned=" | ".join(x for x in (note, validator_note) if x))
        reasons.extend(check_reply(review, rec["reply"]))
    else:
        rec.update(status="fallback", category="other", urgency="medium",
                   summary="Automatic analysis unavailable; generic reply used.",
                   reply=FALLBACK_REPLY, model_reply="", auto_cleaned="")
        detail = f" ({str(result.get('error'))[:160]})" if result.get("error") else ""
        reasons.append(f"LLM {llm_status}: generic reply, review manually{detail}")

    rec["review_reason"] = "; ".join(reasons)
    rec["needs_human_review"] = bool(reasons)
    return rec


# ------------------------------------------------------------------ batch
def _cache_key(base_url: str, model: str, rating, review: str) -> str:
    digest = hashlib.sha1(f"{rating}|{review}".encode("utf-8")).hexdigest()[:16]
    return f"{PROMPT_VERSION}|{base_url}|{model}|{digest}"


def run_pipeline(prepared: pd.DataFrame, api_key: str, base_url: str, model: str,
                 min_interval_s: float = 8.0, cache: dict | None = None,
                 on_progress=None, sleep=None, prices=None) -> tuple[pd.DataFrame, str | None]:
    """Process each review. Returns (results, stop_reason). Stops early if the quota is exhausted.
    Only successful LLM results are cached, so a retry after an error calls the API again."""
    cache = {} if cache is None else cache
    rows, last_call, stop_reason = [], None, None
    total = len(prepared)

    for i, row in enumerate(prepared.itertuples(index=False), start=1):
        key = _cache_key(base_url, model, row.rating, row.review)
        if key in cache:
            rec = dict(cache[key])
            rec.update(case_id=row.case_id, source="cache", latency_s=0.0, cost_usd=0.0,
                       prompt_tokens=0, completion_tokens=0)
        else:
            if last_call is not None:
                wait = min_interval_s - (time.time() - last_call)
                if wait > 0:
                    (sleep or time.sleep)(wait)
            rec = process_review(row.case_id, row.rating, row.review, api_key, base_url, model, prices)
            last_call = time.time()
            rec["source"] = "api"
            if rec["llm_status"] == "ok":
                cache[key] = dict(rec)

        rec["row_id"] = str(i)                         # unique even if the file repeats case ids
        rec["rank"] = getattr(row, "rank", None)
        rec["severity_score"] = getattr(row, "severity_score", None)
        rows.append(rec)
        if on_progress:
            on_progress(i, total, rec)
        if rec["llm_status"] == "quota":
            stop_reason = "quota"
            break

    results = pd.DataFrame(rows)
    if not results.empty:
        results["_u"] = results["urgency"].map(URGENCY_ORDER)
        results = (results.sort_values(["_u", "rank", "row_id"], kind="stable")
                   .drop(columns="_u").reset_index(drop=True))
    return results, stop_reason


OUTPUT_COLUMNS = ["case_id", "rank", "severity_score", "model", "rating", "sentiment", "mismatch", "urgency", "category", "summary",
                  "reply", "edited", "original_reply", "auto_cleaned", "model_reply", "needs_human_review", "review_reason", "status",
                  "error", "review", "latency_s", "prompt_tokens", "completion_tokens", "cost_usd"]


def apply_edits(results: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """Return results with the person's edited replies in `reply` (keyed by row_id)."""
    out = results.copy()
    out["original_reply"] = out["reply"]
    out["edited"] = out["row_id"].map(lambda r: r in edits)
    out["reply"] = [edits.get(r, t) for r, t in zip(out["row_id"], out["reply"])]
    return out


def results_to_csv(results: pd.DataFrame, edits: dict | None = None, approved: set | None = None) -> bytes:
    """CSV of results. If `approved` (a set of row_ids) is given, only those rows are included."""
    data = apply_edits(results, edits or {})
    if approved is not None:
        data = data[data["row_id"].isin(approved)]
    cols = [c for c in OUTPUT_COLUMNS if c in data.columns]
    return data[cols].to_csv(index=False).encode("utf-8-sig")


def list_models(base_url: str, api_key: str, timeout: int = 20) -> tuple[list[str], str | None]:
    """Model ids the provider reports for this key (OpenAI-compatible GET /models)."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/models",
                            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        ids = sorted(str(m.get("id", "")).removeprefix("models/") for m in data if m.get("id"))
        return ids, None
    except Exception as exc:  # network, auth, bad JSON
        return [], f"{type(exc).__name__}: {str(exc)[:200]}"
