"""Streamlit app: upload a reviews CSV, pick the most critical 1-2 star reviews, get a structured triage
and a draft reply for each, edit the replies, and download the final CSV."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import (apply_edits, check_reply, guess_column, list_models, prepare_reviews, rank_reviews,
                      results_to_csv, run_pipeline, select_reviews)

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sample_reviews.csv"
NONE = "(none)"
MAX_PER_RUN = 100          # hard cap so one upload cannot burn a whole quota


def show_table(df):
    """Full-width table that works on old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)            # Streamlit 1.50+
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)   # older versions

st.set_page_config(page_title="Review Response Assistant", layout="wide")
st.title("Review Response Assistant")
st.caption("Upload customer reviews. The app keeps the 1-2 star ones, ranks them by how critical they are, "
           "drafts a reply for the ones you choose, and lets you edit every reply before you download.")

# ------------------------------------------------------------------ sidebar
# Each provider has its own key name, so a key saved for one provider is never sent to another.
PROVIDERS = {
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b",
             "key_name": "GROQ_API_KEY", "price_in": 0.15, "price_out": 0.60,
             "help": "Free key: console.groq.com/keys"},
    "Hugging Face": {"base_url": "https://router.huggingface.co/v1", "model": "openai/gpt-oss-120b",
                     "key_name": "HF_TOKEN", "price_in": 0.0, "price_out": 0.0,
                     "help": "Token needs the 'Make calls to Inference Providers' permission. Free accounts "
                             "get only a small monthly credit."},
    "Google Gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                      "model": "gemini-2.5-flash", "key_name": "GEMINI_API_KEY",
                      "price_in": 0.0, "price_out": 0.0,
                      "help": "Free key: aistudio.google.com/apikey. Use 'List models' to confirm the name."},
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-oss-20b:free",
                   "key_name": "OPENROUTER_API_KEY", "price_in": 0.0, "price_out": 0.0,
                   "help": "Free key: openrouter.ai/keys. Free models end in :free and have tight daily limits."},
    "Custom": {"base_url": "", "model": "", "key_name": "LLM_API_KEY", "price_in": 0.0, "price_out": 0.0,
               "help": "Any OpenAI-compatible endpoint."},
}


def saved_key(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.environ.get(name, "")


with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", list(PROVIDERS))
    cfg = PROVIDERS[provider]
    st.caption(cfg["help"])

    api_key = saved_key(cfg["key_name"])
    if api_key:
        st.success(f"{cfg['key_name']} loaded from secrets / environment")
    else:
        api_key = st.text_input(f"{provider} API key", type="password", key=f"key_{provider}",
                                help="Kept in this session only. Never written to disk. "
                                     f"To save it, add {cfg['key_name']} to .streamlit/secrets.toml")
    base_url = st.text_input("API base URL", cfg["base_url"], key=f"url_{provider}")
    model = st.text_input("Model", cfg["model"], key=f"model_{provider}")

    with st.expander("Check which models my key can use"):
        search = st.text_input("Filter (optional)", key=f"filter_{provider}")
        if st.button("List models", disabled=not (api_key and base_url)):
            ids, err = list_models(base_url, api_key)
            if err:
                st.error(err)
            else:
                shown = [m for m in ids if search.lower() in m.lower()]
                st.write(f"{len(shown)} of {len(ids)} models")
                st.code("\n".join(shown[:80]) or "(none match)")

    price_in = st.number_input("Price per 1M input tokens (USD)", min_value=0.0,
                               value=float(cfg["price_in"]), step=0.05, key=f"pin_{provider}")
    price_out = st.number_input("Price per 1M output tokens (USD)", min_value=0.0,
                                value=float(cfg["price_out"]), step=0.05, key=f"pout_{provider}")
    st.caption("Used only for the cost estimate. Free tiers cost 0.")

    interval = st.slider("Seconds between API calls", 0, 20, 8,
                         help="Free tiers allow only a few calls per minute. Raise this if you see rate-limit errors.")
    st.caption("Review text is sent to the API provider you configure. Do not upload data you are "
               "not allowed to share with them.")

# ------------------------------------------------------------------ input
uploaded = st.file_uploader("Reviews CSV", type="csv")
use_sample = st.checkbox("Use the sample file instead", value=False)

raw = None
if use_sample:
    raw = pd.read_csv(SAMPLE_PATH)
elif uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this file as CSV: {exc}")

if raw is None:
    st.info("Upload a CSV with a review text column and a star rating column, or tick the sample box.")
    st.stop()
if raw.empty:
    st.error("The file has no rows.")
    st.stop()

st.subheader("1. Map your columns")
cols = list(raw.columns)
c1, c2, c3, c4 = st.columns(4)
text_guess = guess_column(cols, ["review_text", "review text", "text", "review", "comment", "body"])
rating_guess = guess_column(cols, ["rating", "star", "score"])
header_guess = guess_column(cols, ["header", "title", "subject", "summary"])
id_guess = guess_column(cols, ["unique_id", "review_id", "id"])

text_col = c1.selectbox("Review text *", cols, index=cols.index(text_guess) if text_guess in cols else 0)
rating_col = c2.selectbox("Star rating *", cols, index=cols.index(rating_guess) if rating_guess in cols else 0)
header_opt = [NONE] + cols
header_col = c3.selectbox("Review title", header_opt, index=header_opt.index(header_guess) if header_guess in cols else 0)
id_col = c4.selectbox("Review ID", header_opt, index=header_opt.index(id_guess) if id_guess in cols else 0)

_kw = dict(header_col=None if header_col == NONE else header_col,
           id_col=None if id_col == NONE else id_col)
_avail = prepare_reviews(raw, text_col, rating_col, **_kw)[1]["mismatch_available"]
include_mismatch = st.checkbox(
    f"Also include 4-5 star reviews whose text is very negative (rating/text mismatch) - {_avail} found",
    value=False, disabled=_avail == 0,
    help="Catches mis-clicked stars, sarcasm, or 'great seller but the product burned me'. Text sentiment is "
         "judged by negative vs positive words, not by the star rating. These replies always need human review.")
prepared, stats = prepare_reviews(raw, text_col, rating_col, include_mismatch=include_mismatch, **_kw)
if prepared.empty:
    st.warning("No 1-2 star reviews found. Check that the rating column is mapped correctly.")
    st.stop()
ranked = rank_reviews(prepared)

# ------------------------------------------------------------------ choose reviews
st.subheader("2. Choose which reviews to generate replies for")
available = min(len(ranked), MAX_PER_RUN)
o1, o2 = st.columns([1, 2])
how_many = o1.number_input(f"How many reviews? (1 to {available})", min_value=1, max_value=available,
                           value=min(5, available), step=1, key=f"n_{len(ranked)}")
order = o2.radio("Pick them by", ["Most critical first", "Order in the file"], horizontal=True)
selected = select_reviews(ranked, how_many, "critical" if order == "Most critical first" else "file")

m = st.columns(6)
m[0].metric("Rows in file", stats["rows_in"])
m[1].metric("Dropped (bad rating / empty / duplicate)",
            stats["dropped_bad_rating"] + stats["dropped_empty_text"] + stats["dropped_duplicates"])
m[2].metric("Not critical (3-5 stars)", stats["not_critical"])
m[3].metric("Critical (1-2 stars)", stats["critical"])
m[4].metric("Rating/text mismatch", stats["mismatch"])
m[5].metric("Selected", len(selected))

with st.expander("Selected reviews and their criticality score", expanded=True):
    show_table(selected[["rank", "case_id", "rating", "mismatch", "severity_score", "word_count", "review"]])
    st.caption("Score = star rating + review detail (capped) + safety, urgency, defect, refund/replacement and "
               "complaint words. Higher means more critical. Rank 1 is the most critical review in the file.")

# ------------------------------------------------------------------ generate
st.subheader("3. Generate")
for key, default in (("cache", {}), ("edits", {}), ("approved", set()), ("run_id", 0)):
    if key not in st.session_state:
        st.session_state[key] = default

if st.button("Generate replies", type="primary", disabled=not (api_key and base_url and model)):
    bar = st.progress(0.0, text="Starting...")

    def on_progress(i, total, rec):
        bar.progress(i / total, text=f"{i}/{total} done (last: case {rec['case_id']}, {rec['status']})")

    results, stop = run_pipeline(selected, api_key, base_url, model, min_interval_s=interval,
                                 cache=st.session_state["cache"], on_progress=on_progress,
                                 prices=(price_in, price_out))
    bar.empty()
    st.session_state["results"] = results
    st.session_state["stop"] = stop
    st.session_state["edits"] = {}                 # a new run starts with no manual edits
    st.session_state["approved"] = set()           # ...and nothing approved
    st.session_state["run_id"] += 1
if not (api_key and base_url and model):
    st.caption("Enter an API key, base URL and model in the sidebar to enable generation.")

# ------------------------------------------------------------------ results + editing
results = st.session_state.get("results")
if results is not None and not results.empty:
    edits = st.session_state["edits"]
    approved = st.session_state["approved"]
    run_id = st.session_state["run_id"]
    if st.session_state.get("stop") == "quota":
        st.warning("The provider's quota ran out, so the run stopped early. Remaining reviews were not processed.")

    st.subheader("4. Review, edit and download")
    k = st.columns(8)
    k[0].metric("Processed", len(results))
    k[1].metric("Need human review", int(results["needs_human_review"].sum()))
    k[2].metric("Auto-cleaned", int((results["auto_cleaned"].fillna("") != "").sum()))
    k[3].metric("Fallback replies", int((results["status"] == "fallback").sum()))
    k[4].metric("Edited by you", len(edits))
    k[5].metric("Approved", f"{len(approved)}/{len(results)}")
    k[6].metric("Avg latency (s)", f"{results.loc[results['source'] == 'api', 'latency_s'].mean():.1f}"
                if (results["source"] == "api").any() else "-")
    k[7].metric("Est. cost (USD)", f"{results['cost_usd'].sum():.4f}")
    st.caption("Auto-cleaned: the app removed promise-like or unsupported sentences from the model's draft and "
               "replaced the 'send your order number' part with a fixed sentence. The original model text "
               "is kept in the CSV (model_reply).")

    only_review = st.checkbox("Show only replies that need human review")
    shown = results[results["needs_human_review"]] if only_review else results
    table = apply_edits(shown, edits)
    table["approved"] = table["row_id"].isin(approved)
    show_table(table[["rank", "case_id", "rating", "sentiment", "mismatch", "urgency", "category", "summary", "approved", "edited",
                      "needs_human_review", "review_reason", "auto_cleaned"]])

    def approve_row(row_id: str, widget_key: str):
        (approved.add if st.session_state[widget_key] else approved.discard)(row_id)

    def approve_shown(value: bool):
        for rid_ in shown["row_id"]:
            st.session_state[f"approve_{run_id}_{rid_}"] = value
            (approved.add if value else approved.discard)(rid_)

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Approve all shown", on_click=approve_shown, args=(True,))
    b2.button("Clear approvals", on_click=approve_shown, args=(False,))

    st.download_button(f"Download approved replies only ({len(approved)}) (CSV)",
                       results_to_csv(results, edits, approved), file_name="approved_review_responses.csv",
                       mime="text/csv", disabled=not approved)
    if not approved:
        st.caption("Tick 'Approve this reply' inside a review below to enable the download.")

    def save_edit(row_id: str, widget_key: str, original: str):
        text = st.session_state[widget_key]
        if text == original:
            st.session_state["edits"].pop(row_id, None)
        else:
            st.session_state["edits"][row_id] = text

    def reset_edit(row_id: str, widget_key: str, original: str):
        st.session_state["edits"].pop(row_id, None)
        st.session_state[widget_key] = original

    st.markdown("**Read each review, edit the reply if needed**")
    st.caption("Click outside the text box (or press Ctrl+Enter) to save an edit. Edits are kept when you "
               "change the filter above, and are included in the download.")
    for _, r in shown.iterrows():
        rid = r["row_id"]
        edited = rid in edits
        tags = ((" - APPROVED" if rid in approved else "") + (" - EDITED" if edited else "")
                + (" - MISMATCH" if r.get("mismatch") else "")
                + (" - NEEDS REVIEW" if r["needs_human_review"] else ""))
        with st.expander(f"#{r['rank']} | Case {r['case_id']} | {str(r['urgency']).upper()} | {r['category']}{tags}"):
            left, right = st.columns(2)
            left.markdown("**Customer review**")
            left.write(r["review"])
            right.markdown("**Draft reply (editable)**")
            wkey = f"reply_{run_id}_{rid}"
            st.session_state.setdefault(wkey, edits.get(rid, r["reply"]))
            right.text_area("Reply", key=wkey, height=300, label_visibility="collapsed",
                            on_change=save_edit, args=(rid, wkey, r["reply"]))
            if r.get("auto_cleaned"):
                right.info("Auto-cleaned: " + str(r["auto_cleaned"]))
            current = st.session_state[wkey]
            if not current.strip():
                right.error("This reply is empty.")
            else:
                issues = check_reply(r["review"], current)
                if r["status"] == "fallback" and not edited:
                    issues = [r["review_reason"]]
                if issues:
                    right.warning("Automatic check: " + " | ".join(issues))
                else:
                    right.success("Automatic check found nothing. Still read it before sending.")
                right.caption(f"{len(current.split())} words")
            akey = f"approve_{run_id}_{rid}"
            st.session_state.setdefault(akey, rid in approved)
            right.checkbox("Approve this reply (included in the download)", key=akey,
                           on_change=approve_row, args=(rid, akey))
            if edited:
                right.button("Reset to the original draft", key=f"reset_{wkey}", on_click=reset_edit,
                             args=(rid, wkey, r["reply"]))
    st.caption("Drafts are suggestions. A person should read each one before it is sent.")
