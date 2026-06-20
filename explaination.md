# BankMind — Customer Subscription Prediction
### Project Explanation

---

## 📌 Required Submission Answers

> *The sections below answer every required question directly. Full reasoning, data, and tables backing each answer are in the numbered sections that follow.*

### Everyone

**1. What percentage of customers have `y = yes`? What does this imbalance mean for evaluation?**
>  **11.7%** of customers subscribed (5,289 of 45,211); the rest (88.3%) did not. This imbalance means **accuracy alone is misleading** — a model predicting "no" for everyone would score ~88% accuracy while catching zero real subscribers. Precision, recall, and F1-score are needed instead. *(Full breakdown: Section 3)*

**2. Which job category had the highest subscription rate? Does this make sense intuitively?**
>  **Student**, at **28.68%**, followed closely by **retired** at **22.79%** — both well above the 11.7% dataset average. Yes, this makes intuitive sense: both groups have more free time for a full sales conversation and fit a low-risk savings product profile (students saving up, retirees protecting capital). *(Full table: Section 3)*

### Track B adds

**1. Which feature had the highest importance in your tree-based model? Why?**
> **`duration`** (call length), at **0.346 importance** — nearly 35% of total importance, more than the next three features combined. It's the top feature because call length is a direct behavioral proxy for engagement: disinterested customers end calls quickly, genuinely interested ones stay on longer. *(Full table: Section 7)*

**2. Why is F1 better than accuracy for this dataset?**
> Because of the 11.7%/88.3% imbalance, accuracy rewards a model for defaulting to the majority "no" class. F1 (the harmonic mean of precision and recall) can't be gamed this way — it forces a balance between catching real subscribers and not wasting RM time on bad leads. F1 is also what correctly identifies **Random Forest (59.49% F1)** as the stronger model overall, even though Logistic Regression has higher raw recall. *(Full comparison: Section 6)*

**3. Pick one sample prediction — do you agree with the model's call?**
> **Customer 43303** (age 42, management, balance ₹1,205) was predicted **YES at 56%** — and the actual outcome was **NO**. I don't agree with this call in hindsight: 56% is barely above the decision threshold, a weak signal rather than a confident one, and it reflects Random Forest's known precision gap (55.79%) — almost half its "yes" calls don't pan out. *(Full walkthrough: Section 8)*

### Track C adds

**1. What would break first with 200 RMs hitting `/predict` simultaneously? What would you change?**
>  Most likely first failure: the **Groq LLM call sitting inside the prediction path** — if `/predict` and `/explain` aren't cleanly separated, every request waits on an external API round-trip, and Groq's rate limits throttle the system before FastAPI itself struggles. **The fix I'd prioritize:** keep `/predict` fully synchronous and self-contained (in-memory model, zero external calls), and move the Groq explanation to its own endpoint. *(Full breakdown: Section 11)*

**2. What does the LLM explanation add over just showing a probability score?**
> It translates a bare number like "78%" into a plain-language reason and a suggested RM approach — business language instead of a stat. **Honest caveat:** it adds no new predictive power, only narrates the model's existing output, and can sound plausible without being strictly grounded in the model's real decision logic if not carefully prompted. *(Full discussion: Section 10)*

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

### `df.info()` output

```
<class 'pandas.DataFrame'>
RangeIndex: 45211 entries, 0 to 45210
Data columns (total 17 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   age        45211 non-null  int64
 1   job        45211 non-null  str
 2   marital    45211 non-null  str
 3   education  45211 non-null  str
 4   default    45211 non-null  str
 5   balance    45211 non-null  int64
 6   housing    45211 non-null  str
 7   loan       45211 non-null  str
 8   contact    45211 non-null  str
 9   day        45211 non-null  int64
 10  month      45211 non-null  str
 11  duration   45211 non-null  int64
 12  campaign   45211 non-null  int64
 13  pdays      45211 non-null  int64
 14  previous   45211 non-null  int64
 15  poutcome   45211 non-null  str
 16  y          45211 non-null  str
dtypes: int64(7), str(10)
memory usage: 5.9 MB
```

Every column shows the full 45,211 non-null count, which confirms the "zero missing values" finding directly from the data rather than just the `isnull().sum()` check later on. Seven numeric columns (`int64`) and ten categorical/text columns — the categorical ones are exactly what get routed through `OneHotEncoder` during preprocessing, and the numeric ones through `StandardScaler`.

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

Actual subscription rate by job category, computed via `pd.crosstab(df['job'], df['y'], normalize='index')`:

| Job | Subscription Rate |
|---|---|
| **Student** | **28.68%** |
| **Retired** | **22.79%** |
| Unemployed | 15.50% |
| Management | 13.76% |
| Admin. | 12.20% |
| Self-employed | 11.84% |
| Unknown | 11.81% |
| Technician | 11.06% |
| Services | 8.88% |
| Housemaid | 8.79% |
| Entrepreneur | 8.27% |
| Blue-collar | 7.28% |

**Students (28.68%)** have the highest subscription rate, followed closely by **retirees (22.79%)** — both well above the dataset's overall average of 11.7%, and roughly 3–4x higher than the lowest category, blue-collar (7.28%).

This tracks intuitively: both groups tend to have more free time to engage in a full conversation (which ties directly into why `duration` turned out to be the top predictive feature — see Section 7), and both often have financial profiles suited to a low-risk, fixed-term product — students saving up, retirees prioritizing capital safety over growth. They're also less likely to be the target of high-pressure, high-volume sales pushes, so when they do engage, the interest tends to be genuine. The balance data backs this up too — subscribers carry a noticeably higher average balance (₹1,804) than non-subscribers (₹1,304), suggesting subscription correlates with having more disposable savings to lock away.

---

## 4. Data Preprocessing

**Categorical:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
**Numerical:** `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`

- Target encoded: `yes → 1`, `no → 0`
- Categorical features one-hot encoded (ML models can't interpret raw text categories)
- A `ColumnTransformer` pipeline applies preprocessing identically at training time and prediction time, preventing train/serve skew
- Stratified 80/20 train-test split, preserving the class ratio in both sets:

| Split | No (0) | Yes (1) | Total |
|---|---|---|---|
| Train | 31,937 | 4,231 | 36,168 |
| Test | 7,985 | 1,058 | 9,043 |

- After encoding, the feature space expands from 16 raw columns to **51 columns** (one-hot encoding the 9 categorical fields multiplies out their categories)

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
| Accuracy | **89.85%** |
| Precision | 55.79% |
| Recall | 63.71% |
| F1 Score | **59.49%** |

Confusion matrix-level detail from the classification report:

```
              precision    recall  f1-score   support
           0       0.95      0.93      0.94      7985
           1       0.56      0.64      0.59      1058
```

Random Forest improves on *every* metric simultaneously compared to Logistic Regression here — higher accuracy, higher precision, and a higher F1, though Logistic Regression still wins on raw recall (81.47% vs. 63.71%).

### The key trade-off

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 84.57% | 41.82% | **81.47%** | 55.27% |
| Random Forest | **89.85%** | **55.79%** | 63.71% | **59.49%** |

Random Forest comes out ahead on **F1 (59.49% vs. 55.27%)**, making it the stronger overall model for this problem — but the trade-off between the two is still real and worth naming explicitly: Logistic Regression catches significantly more actual subscribers (81% recall vs. 64%), at the cost of far more false positives (only 42% precision vs. 56%). In a real deployment, the right choice depends on which error is more expensive for the bank — missing a subscriber (favors Logistic Regression) or wasting an RM's time on a bad lead (favors Random Forest). Random Forest was selected as the main model here because it offers the best *balance* of the two, reflected in its higher F1-score.

---

## 6. Why F1-Score Is the Right Metric Here

With only 11.7% of customers subscribing, accuracy rewards a model for simply favoring the majority class. F1-score is the harmonic mean of precision and recall, so a model can't game it by ignoring the minority class — doing so collapses recall toward zero.

- **Precision** → of everyone flagged as a likely subscriber, how many actually were? (protects RM time)
- **Recall** → of everyone who actually subscribed, how many did the model catch? (protects revenue opportunity)

F1 forces a balance between the two, which is exactly the trade-off this business problem demands — and it's the metric that correctly identifies Random Forest (F1 = 59.49%) as the stronger model overall, even though Logistic Regression wins on recall alone (81.47%) and would look like the better choice if recall were judged in isolation.

---

## 7. Feature Importance Analysis

**Top 10 features (Random Forest), from `feature_importances_`:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `duration` | 0.346083 |
| 2 | `balance` | 0.074402 |
| 3 | `age` | 0.071811 |
| 4 | `day` | 0.065828 |
| 5 | `campaign` | 0.033227 |
| 6 | `pdays` | 0.032711 |
| 7 | `poutcome = success` | 0.029949 |
| 8 | `contact = unknown` | 0.029079 |
| 9 | `housing = yes` | 0.020867 |
| 10 | `previous` | 0.018356 |

`duration` alone accounts for nearly **35% of total importance** — more than the next three features combined — making it by a wide margin the single most decisive signal in the model.

### Why is Duration the top feature?

Call duration is a strong behavioral signal: a disengaged customer ends the call quickly, while genuine interest produces a longer conversation — questions asked, terms discussed, objections worked through. It's the single clearest proxy for engagement in the dataset.

**The catch:** duration is only known *after* the call has already happened. It's highly predictive but not actionable for deciding *whom to call* — it's better suited to post-call scoring than pre-call targeting, and that distinction matters for how the model is eventually deployed.

---

## 8. Sample Prediction Analysis

Five test-set customers (3 predicted YES, 2 predicted NO), pulled directly from the model run:

| Customer ID | Age | Job | Balance | Housing | Loan | Actual | Prediction | Probability |
|---|---|---|---|---|---|---|---|---|
| 43965 | 64 | Retired | 109 | No | No | Yes | **YES** | **98.0%** |
| 43303 | 42 | Management | 1,205 | No | No | No | YES | 56.0% |
| 42761 | 61 | Retired | 89 | No | No | No | YES | 56.0% |
| 1392 | 40 | Blue-collar | 640 | Yes | Yes | No | NO | 0.0% |
| 7518 | 44 | Technician | 378 | Yes | No | No | NO | 0.0% |

### A closer look — do I agree with these calls?

**Customer 43965 (64, retired, balance 109, predicted YES at 98.0%)** is the standout case, and it's the one this section originally focused on. The age and job line up well with the job-category trend in Section 3 — retirees subscribe at roughly double the dataset average. And the *actual* outcome confirms the model was right: this customer really did subscribe.

That said, I'd still flag the same underlying tension: a balance of just 109 is unusually low for someone committing to a term deposit, and `duration` is the model's dominant feature by far (34.6% importance — see Section 7). It's likely this customer had a long, engaged call — possibly reflected in a high `duration` value and a `poutcome = success` from a prior campaign, both of which are heavily weighted features — and that engagement, not the balance, is what's driving the 98% confidence. The model got it right here, but it's a useful reminder that the *reason* behind a correct prediction is still worth checking, not just the outcome.

**Customers 43303 and 42761 (both predicted YES at 56%, both actually NO)** are the more interesting cases for scrutiny. Both sit barely above the 50% decision threshold, and both predictions were **wrong** — the model overestimated their likelihood to subscribe. This is a good illustration of the precision trade-off discussed in Section 5: Random Forest's precision is only 55.79%, meaning almost half of its "yes" predictions don't pan out, and these two examples are exactly that failure mode in action. A 56% probability is a weak signal, not a confident one, and in a real deployment an RM should treat anything this close to the threshold with real caution rather than treating it the same as the 98% case above.

**Customers 1392 and 7518 (both predicted NO at 0.0%, both actually NO)** are the easy, confidently-correct cases — both have housing and/or personal loans, which the feature importance table shows the model weighs meaningfully (`housing = yes` is in the top 10). Existing debt obligations make a new term deposit commitment less likely, and the model's near-zero probability reflects that cleanly.

**Overall takeaway:** the model performs well at the extremes (very high or very low probability) but is noticeably less reliable in the middle band around 50–60%, which is exactly where precision and recall trade off against each other — and exactly where a relationship manager's own judgment should weigh more heavily than the model's score.

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

Beyond just producing a prediction, the project reflects a deliberate set of decisions — choosing F1 over accuracy because of class imbalance, selecting Random Forest because it wins on F1 despite Logistic Regression's edge in raw recall, treating `duration` as informative but not actionable, and scrutinizing individual predictions (including the misfires in the mid-50% range) rather than trusting aggregate metrics blindly. The result is a system that doesn't just output a probability, but supports a relationship manager in making sense of — and acting on — that number.
