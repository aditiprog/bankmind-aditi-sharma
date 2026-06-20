# BankMind — Customer Subscription Prediction
### Project Explanation

---

## 1. Project Overview

BankMind is a machine learning system built on the **UCI Bank Marketing Dataset** to predict whether a customer will subscribe to a term deposit after a marketing campaign.

The dataset holds real records from a Portuguese bank — demographics, financial standing, existing products, campaign history, and the final outcome of each contact.

**Pipeline:** EDA → Preprocessing → Model Comparison → Evaluation → Feature Importance → FastAPI Deployment → LLM-Based Explanation (Groq).

---

## 2. Dataset Understanding

**45,211 records · 17 columns**

| Feature | Description |
|---|---|
| `age` | Customer age |
| `job` | Occupation |
| `education` | Education level |
| `balance` | Yearly average account balance |
| `housing` | Existing housing loan |
| `loan` | Existing personal loan |
| `campaign` | Contacts made during this campaign |
| `previous` | Prior campaign interactions |
| `poutcome` | Outcome of the previous campaign |

**Target (`y`):** `yes` → subscribed · `no` → did not subscribe

---

## 3. Exploratory Data Analysis

**Missing values:** None across any column — no imputation required.

**Class distribution:**
- No subscription: 39,922
- Subscription: 5,289
- **Only 11.7% of customers subscribed.**

### Why this matters

The dataset is heavily imbalanced — the vast majority of customers say no. This single fact shapes everything downstream:

> **A model that predicts "no" for every customer would score ~88% accuracy while being completely useless.** It would never identify a single real subscriber.

Because of this, **accuracy alone cannot be trusted**. Precision, recall, and F1-score became the metrics that actually mattered for judging whether the model adds business value.

### Job category and subscription rate

On this dataset, **students** and **retired** customers consistently show the highest subscription rates (often 20–28%), well above categories like blue-collar or entrepreneur (often under 10%).

This tracks intuitively: both groups tend to have more time to engage in a full conversation (which ties directly into why `duration` turned out to be the top predictive feature — see Section 7), and both often have financial profiles suited to a low-risk, fixed-term product — students saving up, retirees prioritizing capital safety over growth. They're also less likely to be the target of high-pressure, high-volume sales pushes, so when they do engage, the interest tends to be genuine.

---

## 4. Data Preprocessing

**Categorical:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
**Numerical:** `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`

- Target encoded: `yes → 1`, `no → 0`
- Categorical features one-hot encoded (ML models can't interpret raw text categories)
- A `ColumnTransformer` pipeline applies preprocessing identically at training time and prediction time, preventing train/serve skew

---

## 5. Model Training

### Model 1 — Logistic Regression (Baseline)

Chosen for simplicity, interpretability, and clean probability outputs.

| Metric | Score |
|---|---|
| Accuracy | 84.57% |
| Precision | 41.82% |
| Recall | **81.47%** |
| F1 Score | **55.27%** |

High recall — it catches most real subscribers, at the cost of more false positives.

### Model 2 — Random Forest (Main Model)

Chosen to capture nonlinear relationships and feature interactions, and to expose feature importances.

| Metric | Score |
|---|---|
| Accuracy | **90.57%** |
| Precision | **69.38%** |
| Recall | 34.69% |
| F1 Score | 46.25% |

Higher accuracy and precision, but notably **lower recall** — it misses more than half the real subscribers.

### The key trade-off

Random Forest *looks* better on accuracy alone (90.57% vs. 84.57%) — but its F1-score is actually **lower** than Logistic Regression's (46.25% vs. 55.27%). Since the cost of missing a real subscriber outweighs the cost of a few wasted calls, **accuracy alone would have pointed to the wrong model.** This is the clearest evidence in the project for why a single metric isn't enough.

---

## 6. Why F1-Score Is the Right Metric Here

With only 11.7% of customers subscribing, accuracy rewards a model for simply favoring the majority class. F1-score is the harmonic mean of precision and recall, so a model can't game it by ignoring the minority class — doing so collapses recall toward zero.

- **Precision** → of everyone flagged as a likely subscriber, how many actually were? (protects RM time)
- **Recall** → of everyone who actually subscribed, how many did the model catch? (protects revenue opportunity)

F1 forces a balance between the two, which is exactly the trade-off this business problem demands.

---

## 7. Feature Importance Analysis

**Top features (Random Forest):**
1. **Duration**
2. Balance
3. Age
4. Day
5. Previous campaign outcome

### Why is Duration the top feature?

Call duration is a strong behavioral signal: a disengaged customer ends the call quickly, while genuine interest produces a longer conversation — questions asked, terms discussed, objections worked through. It's the single clearest proxy for engagement in the dataset.

**The catch:** duration is only known *after* the call has already happened. It's highly predictive but not actionable for deciding *whom to call* — it's better suited to post-call scoring than pre-call targeting, and that distinction matters for how the model is eventually deployed.

---

## 8. Sample Prediction Analysis

**Example customer:**

| Field | Value |
|---|---|
| Age | 64 |
| Job | Retired |
| Balance | 109 |

**Model output:** `YES` — **83.5% probability**

### A closer look — do I agree with this call?

Partially, and it's worth walking through why.

The **age and job** support the prediction well — retired customers skew toward subscribing, consistent with the job-category trend in Section 3. That part checks out.

The **balance of 109** is where I'd push back. That's a very low balance for someone to commit to a term deposit, which typically requires locking away a meaningful sum. Since `duration` is the model's dominant feature, there's a real risk the high probability is being driven by call length rather than actual capacity to subscribe — for example, a long call full of questions from someone who ultimately can't afford the product. Before trusting this prediction at face value, I'd want to check this customer's `duration` and `poutcome` values directly. It's a useful reminder that feature importance at the model level doesn't guarantee every individual prediction is well-reasoned — predictions still need to be sanity-checked case by case.

---

## 9. FastAPI System

Trained model persisted as `customer_subscription_model.pkl` and served through a FastAPI app.

### `GET /health`
Confirms the API is live.
```json
{ "status": "ok", "model": "Random Forest" }
```

### `POST /predict`
Accepts customer data as JSON → runs it through the same preprocessing pipeline → returns the model's prediction.
```json
{ "will_subscribe": true, "probability": 0.78 }
```

---

## 10. Groq LLM Explanation Feature

A bonus endpoint converts the raw model output into a human-readable explanation — rather than just "probability = 78%," it produces a short narrative on *why* the customer looks promising and *how* a relationship manager should approach them.

### What does this actually add over a bare probability score?

It bridges the gap between a number and an action:
- **Translates output into business language** — e.g. "this customer shows prior positive engagement and a healthy balance, similar to profiles that historically convert."
- **Suggests next steps** — frames the score as guidance for the RM's actual conversation, not just a stat.
- **Builds trust** — non-technical staff are often skeptical of unexplained black-box numbers; a plain-language rationale makes the score easier to act on.

**The honest caveat:** the LLM isn't adding new predictive power — it's narrating the model's existing output and feature values. If not carefully prompted, it can also produce explanations that sound plausible but aren't strictly grounded in the model's actual decision logic. That's a known limitation of layering LLM explanations on top of black-box models, and worth stating explicitly rather than glossing over.

---

## 11. Scaling Considerations — 200 RMs Hitting `/predict` Simultaneously

If 200 relationship managers hit the endpoint at once, a few things would likely break first, roughly in order of impact:

1. **Single-worker server** — if `uvicorn` is running with one worker (typical in a default setup), requests queue up and latency spikes sharply, since each call blocks on model inference.
2. **Model reloaded per request** — if the `.pkl` isn't cached in memory at startup, reloading it from disk on every call multiplies latency under load.
3. **The Groq LLM call inside the prediction path** — if `/predict` waits synchronously on an external API call for the explanation, that network round-trip is by far the slowest part of the request, and external rate limits could throttle the system before the server itself does.
4. **Blocking I/O inside an async endpoint** — synchronous calls inside `async def` block the whole event loop instead of letting concurrent requests proceed.

**The one change I'd prioritize:** decouple the LLM explanation from the core prediction. Keep `/predict` fast and synchronous — in-memory model, no external calls — and move the Groq explanation to its own endpoint or an async follow-up the RM can request after the fact. That removes the biggest bottleneck from the critical path. Running the app with multiple workers and loading the model once at startup (instead of per-request) would handle the rest of the concurrent load.

---

## 12. Limitations

- Trained on a single bank's data — may not generalize to other institutions or regions
- `Duration` is highly predictive but only known *after* the call, limiting its use for pre-call targeting
- Customer behavior shifts over time; the model reflects a snapshot, not a moving target
- No real-time macroeconomic signals are incorporated

---

## 13. Future Improvements

- Hyperparameter tuning
- Stronger models (XGBoost, CatBoost)
- Post-deployment monitoring
- Automated testing
- Dashboard integration
- SHAP-based explainability for per-prediction transparency

---

## Conclusion

BankMind demonstrates a complete, end-to-end ML workflow: **data analysis → preprocessing → model training → evaluation → interpretation → API deployment → AI-powered explanation.**

Beyond just producing a prediction, the project reflects a deliberate set of decisions — choosing F1 over accuracy because of class imbalance, recognizing that Random Forest's higher accuracy actually hid a *worse* F1-score, treating `duration` as informative but not actionable, and scrutinizing individual predictions rather than trusting aggregate metrics blindly. The result is a system that doesn't just output a probability, but supports a relationship manager in making sense of — and acting on — that number.
