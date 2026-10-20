# customer_review_reply_assistant
**Live demo:** https://review-reply-assistant.streamlit.app
The demo needs your own free API key (Groq, Hugging Face or Gemini). Paste it in the sidebar. It is used only for your session.

# Customer Review Reply Assistant

**Live demo:** https://review-reply-assistant.streamlit.app
The demo needs your own free API key (Groq, Hugging Face or Gemini). Paste it in the sidebar; it is used only for your session.

An LLM pipeline that finds the most critical negative customer reviews, classifies each complaint, rates its urgency and drafts a safe reply. A person edits and approves each reply before anything is downloaded.

## 1. Problem and use case

Support teams receive more negative reviews than they can read, and a generic "sorry for the inconvenience" template ignores what the customer actually said. This tool helps a support agent:

1. Pick the N most critical reviews (for example the top 3, 4 or 10).
2. See a category, a one-line summary and an urgency level for each.
3. Start from a draft reply that refers to the customer's real problem, edit it, approve it, and export only the approved replies.

It produces drafts for a human. It does not send anything.

## 2. Dataset and model details

**Dataset.** An Amazon product-review CSV with 60,889 rows and the columns `Unique_ID, Category, Review_Header, Review_text, Rating, Own_Rating`. Rating counts: 5 stars 34,460; 4 stars 12,976; 3 stars 4,366; 1 star 6,979; 2 stars 2,108. The pipeline keeps 1-2 star reviews (8987 rows) after dropping empty, duplicate and badly rated rows. The full file is not in this repository (size and redistribution terms); `sample_data/sample_reviews.csv` is a 26-row sample taken from it.

**Model.** Any OpenAI-compatible chat model. Default: `openai/gpt-oss-120b` on Groq. Others tested through the same interface: Hugging Face router, Gemini, OpenRouter, custom endpoint. The model returns JSON with `category`, `summary`, `urgency` and `response`.

**Limitations.**
- Urgency comes from the model and was **not** validated against human labels.
- Evaluation used small samples (30 reviews for quality and cost, 10 for urgency). Treat the numbers as a demo, not a benchmark.
- An LLM judge from the same model family was tried and found unreliable, so it is not used for scoring.
- The reply sanitizer removes known promise patterns and some invented context. It cannot catch every invented detail, so a person must read each draft.
- Rating/text mismatch detection is a word-list heuristic and can miss sarcasm.
- Cost figures use Groq prices by default. Output quality depends on the model chosen.

## 3. Architecture

```
CSV upload -> column mapping -> clean (empty, duplicate, bad rating)
  -> keep 1-2 star reviews (optional: 4-5 star reviews with very negative text)
  -> severity score -> rank -> top N
  -> LLM call (JSON mode) -> validate -> one repair call -> salvage if only wording rules failed
  -> reply sanitizer (drop promises and unrequested remedies, fixed order-number request)
  -> flags for human review -> edit -> approve -> download approved CSV
```

Design choices:
- **Structured output with validation.** Category and urgency must come from fixed lists; invalid output gets one repair attempt, then a generic fallback reply that is flagged.
- **Replies are rebuilt after generation.** Telling the model "never promise" lowered but did not remove phrases like "we will resolve this" or "keep you updated", so the app removes them and uses a fixed sentence to ask for the order number.
- **Severity score.** Rule-based and explainable: star rating, review detail, safety, defect, refund and complaint words.
- **Rate-limit aware.** Retries on 429 using the provider's wait time; stops cleanly when the quota is exhausted; successful results are cached.

## 4. Evaluation results

Measured in `customer_review_responses_openai.ipynb` on 30 reviews (20 detailed, 10 short), model `openai/gpt-oss-120b`, before the reply sanitizer was added:

| Measure | Result |
|---|---|
| Valid structured output (first run) | 29/30 (97%) |
| Delivered replies passing format and safety checks | 28/29 |
| Latency | median 5.9 s, p95 13.2 s |
| Cost | about $0.00145 per reply ($1.45 per 1,000), 88% of it output tokens |
| Urgency, LLM vs rule-based baseline | LLM higher on 17, same on 12, lower on 0 |
| Urgency agreement on the 10 most critical reviews | 4/10 with the first prompt, 6/10 after adding calibration examples |

Notes:
- The rule baseline rated 25 of 29 reviews `low`, so its thresholds are conservative. The agreement figure is not an error rate; disagreements are flagged for human review.
- The one invalid output was a 61-word reply to a very short review.
- Deterministic checks (unrequested remedies, soft promises) replaced the LLM judge, which scored its own family's output too generously.
- Edit the numbers above if you re-run the notebook with other data.

## 5. Sample output

`sample_data/sample_reviews.csv` contains 26 reviews. To see the app work, tick "Use the sample file instead".

Example of the sanitizer on a **5-star review with a very negative text** (offline test with a mocked model draft, not a live model run):

| | |
|---|---|
| Review | "Great seller but the product is garbage. The plug sparked and burned my hand. Total waste of money, I want a refund." |
| Flags | MISMATCH (5 stars, very negative text), needs human review |
| Model draft removed | "Thank you for your great rating", "We will fix this quickly" |
| Final reply | We are sorry the product sparked and burned your hand. Please reply with your order number or invoice so our support team can review whether this request is eligible. If your star rating does not reflect your experience, you are welcome to update it. |

## 6. Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Put your key in `.streamlit/secrets.toml` (copy `secrets.toml.example`) or paste it in the sidebar. Never commit `secrets.toml`.

Deploy: push to GitHub, create an app on Streamlit Community Cloud with `app.py` as the main file, and add the key under Secrets (or leave it empty so visitors bring their own).

## 7. Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface: upload, mapping, selection, generate, edit, approve, download |
| `pipeline.py` | Cleaning, severity ranking, sentiment mismatch, LLM run, reply sanitizer, CSV export |
| `llm_structured.py` | Prompt, JSON validation, retry and API call |
| `eval_tools.py` | Evaluation helpers (grounding checks, summaries) |
| `customer_review_responses_openai.ipynb` | Development and evaluation notebook |
| `sample_data/sample_reviews.csv` | 26-review sample |

## 8. Future improvements

- Validate urgency against human-labelled data and tune the thresholds.
- Evaluate reply quality with human ratings, not only automatic checks.
- Add a model-based sentiment label next to the word-list check.
- Compare several models on the same reviews (quality, latency, cost).
- Support more languages and mixed-language reviews.
- Connect to a helpdesk or email tool for sending approved replies.
- Add automated tests and CI.

