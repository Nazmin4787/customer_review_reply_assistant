"""Step 7 helpers: LLM-as-judge quality scores, a human rating sheet, and the final summary table.

Self-contained on purpose: it does not import llm_structured, so it cannot disturb your v4 prompt.

Notebook usage:
    import importlib, eval_tools; importlib.reload(eval_tools)
    from eval_tools import judge_replies, make_rating_sheet, build_summary, summarize_human_check
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

JUDGE_VERSION = "j2"
CRITERIA = ["relevance", "personalization", "professionalism", "factual_safety"]
LEVELS = ["low", "medium", "high"]
RANK = {lvl: i for i, lvl in enumerate(LEVELS)}

JUDGE_PROMPT = """You are a strict QA reviewer for customer-support email drafts.
You get ONE customer review and ONE draft reply. Score the reply from 1 (very poor) to 5 (excellent)
on each criterion. Be strict: give 5 only when nothing could be improved.

FIXED DESIGN RULES (do NOT penalize these):
- The greeting "Dear Valued Customer," is REQUIRED when the review gives no customer name.
- The reply must mention at most TWO problems; do not penalize omitting the others.
- The reply must only ask for an order number/invoice; it must not promise any outcome.
- Only judge against what is actually written in the review. Never claim the review mentions
  something unless you can see it there.

relevance        5 = answers the customer's MAIN complaint directly; 1 = off-topic.
personalization  5 = names the customer's specific problem(s) using details from the review;
                 3 = partly specific; 1 = generic text that could answer any review. A very short
                 review (under 15 words) cannot support more than its own details: judge fairly.
professionalism  5 = calm, warm, clear, correct grammar, no awkward phrasing, no filler;
                 1 = rude or sloppy. Deduct for typos or awkward wording.
factual_safety   Do this check first: list every claim, request or detail in the reply that is NOT
                 in the review (for example a refund/replacement/repair the customer never asked
                 for, an invented purpose or context, a mention of a website/app/phone number,
                 a promise to investigate or to find a solution, an invented date or update).
                 5 = none found; 4 = one trivial item; 3 = one clear unsupported item or soft
                 promise; 2 = two or more; 1 = invents order numbers, policies, compensation,
                 or admits liability.

Return ONLY JSON with exactly these keys:
{"unsupported": "the unsupported items you found, or none",
 "relevance": int, "personalization": int, "professionalism": int, "factual_safety": int,
 "reason": "one sentence, max 30 words, naming the main weakness"}
"""


# ------------------------------------------------------------------ low-level call
def _retry_delay(text: str, attempt: int) -> float:
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)", text)
    if m:
        v = float(m.group(1))
        return (v / 1000 if m.group(2) == "ms" else v) + 1.0
    return min(2 ** attempt * 3, 60)


def _fatal_quota(status: int, text: str) -> bool:
    t = text.lower()
    return (status == 402 or "insufficient_quota" in t or "tokens per day" in t
            or "per day" in t or "exceeded your current quota" in t)


def _chat(api_key, base_url, model, system, user, reasoning_effort=None, max_attempts=5):
    """One chat call with 429-aware retries. Returns dict(status, content, error, latency_s, tokens)."""
    out = {"status": "error", "content": "", "error": None, "latency_s": 0.0,
           "prompt_tokens": 0, "completion_tokens": 0}
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "response_format": {"type": "json_object"}}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    url = f"{(base_url or 'https://api.openai.com/v1').rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    start = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as exc:
            out["error"] = f"network: {exc}"
            time.sleep(_retry_delay("", attempt))
            continue
        if resp.status_code >= 400:
            text = resp.text[:600]
            out["error"] = f"{resp.status_code}: {text}"
            if _fatal_quota(resp.status_code, text):
                out["status"] = "quota"
                break
            if resp.status_code == 400 and "response_format" in payload:
                payload.pop("response_format")
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(_retry_delay(text, attempt))
                continue
            break
        body = resp.json()
        usage = body.get("usage", {})
        out["prompt_tokens"] = usage.get("prompt_tokens", 0)
        out["completion_tokens"] = usage.get("completion_tokens", 0)
        out["content"] = (body["choices"][0]["message"].get("content") or "").strip()
        out.update(status="ok", error=None)
        break
    out["latency_s"] = round(time.perf_counter() - start, 2)
    return out


def _parse_scores(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        data = json.loads(m.group(0))
    scores = {}
    for c in CRITERIA:
        v = int(data[c])
        if not 1 <= v <= 5:
            raise ValueError(f"{c} out of range: {v}")
        scores[c] = v
    scores["reason"] = str(data.get("reason", ""))[:300]
    scores["unsupported"] = str(data.get("unsupported", ""))[:300]
    return scores


# ------------------------------------------------------------------ LLM judge
def judge_replies(df: pd.DataFrame, api_key: str, base_url: str | None, model: str,
                  id_col: str = "case_id", review_col: str = "customer_message",
                  reply_col: str = "response", status_col: str = "status",
                  cache_path: str | Path = "outputs/judge_cache.json",
                  min_interval_s: float = 6.0, reasoning_effort: str | None = "low",
                  verbose: bool = True) -> pd.DataFrame:
    """Score every delivered reply (status == ok) on the four criteria. Cached, so re-runs are free.
    Returns df with judge_<criterion>, judge_reason, judge_status columns added."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    prefix = f"{JUDGE_VERSION}|{model}|"

    result_rows, last_call = {}, 0.0
    todo = df[df[status_col] == "ok"]
    for n, (idx, row) in enumerate(todo.iterrows(), start=1):
        key = prefix + str(row[id_col])
        if key in cache and cache[key].get("status") == "ok":
            result_rows[idx] = cache[key]
            source = "cache"
        else:
            wait = min_interval_s - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            user = (f"--- CUSTOMER REVIEW ---\n{str(row[review_col]).strip()[:3500]}\n"
                    f"--- DRAFT REPLY ---\n{str(row[reply_col]).strip()}")
            call = _chat(api_key, base_url, model, JUDGE_PROMPT, user, reasoning_effort)
            last_call = time.time()
            source = "api"
            rec = {"status": call["status"], "error": call["error"], "judge_latency_s": call["latency_s"],
                   "judge_prompt_tokens": call["prompt_tokens"],
                   "judge_completion_tokens": call["completion_tokens"]}
            if call["status"] == "ok":
                try:
                    rec.update(_parse_scores(call["content"]))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    rec.update(status="invalid", error=f"bad judge output: {exc}")
            result_rows[idx] = rec
            if rec["status"] in ("ok", "invalid"):
                cache[key] = rec
                cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            if rec["status"] == "quota":
                print("Daily quota exhausted: stopping. Re-run later; cached scores are kept.")
                break
        if verbose:
            print(f"[{n}/{len(todo)}] case {row[id_col]}: {result_rows[idx]['status']} ({source})")

    out = df.copy()
    for c in CRITERIA:
        out[f"judge_{c}"] = pd.Series({i: r.get(c) for i, r in result_rows.items()}, dtype="float")
    out["judge_reason"] = pd.Series({i: r.get("reason") for i, r in result_rows.items()}, dtype="object")
    out["judge_unsupported"] = pd.Series({i: r.get("unsupported") for i, r in result_rows.items()}, dtype="object")
    out["judge_status"] = pd.Series({i: r["status"] for i, r in result_rows.items()}, dtype="object")
    for col in ("judge_latency_s", "judge_prompt_tokens", "judge_completion_tokens"):
        out[col] = pd.Series({i: r.get(col) for i, r in result_rows.items()}, dtype="float")
    return out


# ------------------------------------------------------------------ automatic grounding check
# A reply may only mention a remedy if the customer's review does. Asking for an order number
# or invoice is part of the template, so it is not checked. This is a FLAG for a human, not a verdict.
_REPLY_TERMS = {"refund": r"refund", "replacement": r"replac|exchange", "repair": r"repair",
                "return": r"\breturn", "warranty": r"warrant"}
_REVIEW_TERMS = {"refund": r"refund|money back|chargeback",
                 "replacement": r"replac|exchange|new (piece|peice|one|unit|phone|device|product)",
                 "repair": r"repair|fix|service cent",
                 "return": r"return|pick ?up", "warranty": r"warrant|guarantee"}


def grounding_flags(review: str, reply: str) -> list[str]:
    """Remedy words used in the reply that the review never mentions."""
    review, reply = str(review).lower(), str(reply).lower()
    return [t for t, pat in _REPLY_TERMS.items()
            if re.search(pat, reply) and not re.search(_REVIEW_TERMS[t], review)]


def add_grounding_flags(df: pd.DataFrame, review_col: str = "customer_message",
                        reply_col: str = "response") -> pd.DataFrame:
    out = df.copy()
    out["grounding_flags"] = [grounding_flags(r, a) if isinstance(a, str) else []
                              for r, a in zip(out[review_col], out[reply_col])]
    return out


# ------------------------------------------------------------------ human rating sheet
def make_rating_sheet(df: pd.DataFrame, path: str | Path, review_col: str = "customer_message",
                      n_total: int = 12, n_lowest: int = 5, seed: int = 7) -> pd.DataFrame:
    """Write a CSV for manual checking: the n_lowest judge-scored replies + random others.
    Open in Excel, fill the 5 blank columns, save as rating_sheet_filled.csv in the same folder."""
    ok = df[df["status"] == "ok"].copy()
    if "judge_relevance" in ok.columns and ok["judge_relevance"].notna().any():
        ok["_min"] = ok[[f"judge_{c}" for c in CRITERIA]].min(axis=1)
        lowest = ok.sort_values("_min").head(n_lowest)
    else:
        lowest = ok.head(0)
    rest = ok.drop(lowest.index)
    rest = rest.sample(min(len(rest), max(n_total - len(lowest), 0)), random_state=seed)
    pick = pd.concat([lowest, rest])
    sheet = pd.DataFrame({
        "case_id": pick["case_id"].astype(str).values,
        "review": pick[review_col].astype(str).str.slice(0, 1200).values,
        "summary": pick["summary"].values,
        "llm_category": pick["category"].values,
        "llm_urgency": pick["urgency"].values,
        "reply": pick["response"].values,
        "auto_flags": (pick["grounding_flags"].map(lambda x: ", ".join(x)).values
                       if "grounding_flags" in pick.columns else ""),
        "relevance_1to5": "", "personalization_1to5": "", "professionalism_1to5": "",
        "factual_safety_1to5": "", "human_urgency": "",
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    return sheet


def summarize_human_check(eval_df: pd.DataFrame, filled_path: str | Path) -> pd.DataFrame:
    """Compare human labels with the judge (quality) and with LLM / rules / hybrid (urgency)."""
    h = pd.read_csv(filled_path, encoding="utf-8-sig")
    h["case_id"] = h["case_id"].astype(str)
    m = eval_df.assign(case_id=eval_df["case_id"].astype(str)).merge(h, on="case_id", suffixes=("", "_h"))
    rows = []
    for c in CRITERIA:
        col = f"{c}_1to5"
        hv = pd.to_numeric(m[col], errors="coerce")
        row = {"criterion": c, "n_rated": int(hv.notna().sum()), "human_mean": round(hv.mean(), 2)}
        if f"judge_{c}" in m.columns:
            diff = (m[f"judge_{c}"] - hv).abs()
            row["judge_mean_same_rows"] = round(m.loc[hv.notna(), f"judge_{c}"].mean(), 2)
            row["judge_within_1_of_human"] = f"{(diff <= 1).sum()}/{int(diff.notna().sum())}"
        rows.append(row)
    quality = pd.DataFrame(rows)

    human = m["human_urgency"].str.strip().str.lower()
    valid = human.isin(LEVELS)
    m = m[valid].copy()
    m["hybrid_urgency"] = [max(a, b, key=RANK.get) for a, b in zip(m["urgency"], m["baseline_urgency"])]
    urg = []
    for name, col in (("LLM", "urgency"), ("Rules", "baseline_urgency"), ("Hybrid (max)", "hybrid_urgency")):
        gap = m[col].map(RANK) - human[valid].map(RANK)
        urg.append({"method": name, "n": len(m), "exact": f"{(gap == 0).mean():.0%}",
                    "within_1": f"{(gap.abs() <= 1).mean():.0%}", "under_called": f"{(gap < 0).mean():.0%}",
                    "over_called": f"{(gap > 0).mean():.0%}"})
    return quality, pd.DataFrame(urg)


# ------------------------------------------------------------------ final summary
def build_summary(eval_df: pd.DataFrame, out_csv: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    """Final evaluation table. Needs columns: status, latency_s, prompt_tokens, completion_tokens,
    cost_usd, passes_checks, urgency, baseline_urgency (and judge_* if you ran the judge)."""
    n = len(eval_df)
    ok = eval_df[eval_df["status"] == "ok"]
    lat = eval_df["latency_s"].dropna()
    rows = [
        ("Reviews evaluated", n),
        ("Valid structured output (status ok)", f"{len(ok)}/{n} ({len(ok) / n:.0%})"),
        ("Delivered replies passing format/safety checks", f"{int(ok['passes_checks'].sum())}/{len(ok)}"),
        ("Latency per review: median / p95 / max (s)",
         f"{lat.median():.1f} / {lat.quantile(0.95):.1f} / {lat.max():.1f}"),
        ("Tokens per review: prompt / completion (mean)",
         f"{eval_df['prompt_tokens'].mean():.0f} / {eval_df['completion_tokens'].mean():.0f}"),
        ("Cost per reply (USD)", f"{eval_df['cost_usd'].mean():.5f}"),
        ("Cost per 1,000 replies (USD)", f"{1000 * eval_df['cost_usd'].mean():.2f}"),
    ]
    gap = eval_df["urgency"].map(RANK) - eval_df["baseline_urgency"].map(RANK)
    rows.append(("LLM urgency vs rules: higher / same / lower",
                 f"{int((gap > 0).sum())} / {int((gap == 0).sum())} / {int((gap < 0).sum())}"))
    if "grounding_flags" in eval_df.columns:
        flagged = eval_df.loc[eval_df["status"] == "ok", "grounding_flags"].map(len) > 0
        rows.append(("Replies mentioning a remedy the review never asked for (auto flag)",
                     f"{int(flagged.sum())}/{len(ok)}"))
    if "judge_relevance" in eval_df.columns:
        j = eval_df[eval_df["judge_status"] == "ok"]
        for c in CRITERIA:
            rows.append((f"LLM-judge {c} (mean of {len(j)}, 1-5)", f"{j[f'judge_{c}'].mean():.2f}"))
        allgood = (j[[f"judge_{c}" for c in CRITERIA]] >= 4).all(axis=1)
        rows.append(("Replies scored >= 4 on all four criteria", f"{int(allgood.sum())}/{len(j)}"))
    summary = pd.DataFrame(rows, columns=["metric", "value"])
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv, index=False)
    md = "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
    return summary, md
