# 🍽️ Order Cancellation Risk Predictor

A professional Streamlit dashboard built around a **pre-trained, frozen**
scikit-learn Logistic Regression pipeline. It scores food-delivery orders
for cancellation risk in real time, explains *why* an order is risky using
the model's own coefficients, and turns that explanation into concrete
next-best-actions for the operations desk.

> **The model is never retrained here.** All preprocessing (imputation,
> scaling, one-hot encoding) lives inside the saved pipeline — this app only
> ever passes raw feature values into `pipeline.predict_proba()`.

---

## Contents

- [Project files](#project-files)
- [Setup & run](#setup--run)
- [What's inside](#whats-inside)
- [Design](#design)
- [How the recommendations work](#how-the-recommendations-work)
- [One assumption worth checking](#one-assumption-worth-checking)
- [Compatibility note](#compatibility-note)
- [Customizing](#customizing)
- [Troubleshooting](#troubleshooting)

---

## Project files

Keep all of these in **the same folder** — the app loads everything by a
path relative to `app.py` itself, regardless of where you launch it from:

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (all logic and UI) |
| `requirements.txt` | Python dependencies |
| `food_delivery_logistic_model.pkl` | Trained scikit-learn pipeline |
| `best_threshold.pkl` | Tuned decision threshold for the "cancel" class |
| `orders.csv` | Historical orders (used for batch merge & insights) |
| `customers.csv` | Customer lookup (type, area, order history) |
| `restaurants.csv` | Restaurant lookup (rating, prep time, cuisine, area) |

## Setup & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. No environment
variables or external services are needed — everything runs locally off
the files in the project folder.

## What's inside

The app has three tabs:

### 🔮 Single Prediction
Pick a customer/restaurant from your data (or enter details manually),
fill in the order specifics, and get:
- a **probability gauge** plus a clear risk verdict against the saved threshold
- a **"what's driving this score"** chart, built directly from the logistic
  regression's own coefficients — no black box, this is the exact math
  behind the prediction
- **recommended actions** for high-risk orders in plain language
  (e.g. *"COD payment — consider an OTP confirmation call"*), or the
  **protective factors** keeping a low-risk order safe

### 📄 Batch Prediction
Upload a CSV of new orders (same shape as `orders.csv`), score them all at
once, see the **top recurring risk drivers** across the flagged orders, and
download the full results as CSV.

### 📊 Portfolio Insights
Cancellation-rate breakdowns from your historical `orders.csv`: payment
method, rider availability, hour of day, delivery distance, restaurant
area, and cuisine — with a quick-read summary of the biggest hotspots.

## Design

A classic navy / cream / gold theme throughout, with `Playfair Display`
headers and a dark navy sidebar for model documentation. The risk verdict
card uses a deep red gradient for high-risk orders and a deep green
gradient for low-risk orders, so the outcome is unmistakable at a glance.
Sidebar reference material (model info, known assumptions, data counts) is
laid out as scannable bullet points rather than paragraphs.

## How the recommendations work

The saved model is a Logistic Regression, which is linear in its (scaled
and encoded) inputs — so `value × coefficient` for each feature gives an
**exact, faithful** breakdown of how much that feature pushed a specific
order's prediction up or down. The app:

1. Transforms the order through the pipeline's own preprocessor
2. Multiplies each transformed feature by its fitted coefficient
3. Ranks the results to find the biggest positive (risk-increasing) and
   negative (risk-reducing) contributors
4. Maps the top ones to an operational playbook — rider allocation, payment
   confirmation, ETA communication, etc.

That mapping lives in `NUMERIC_DRIVER_COPY` and `CATEGORICAL_DRIVER_COPY`
near the top of `app.py`. Edit those dictionaries to match your team's
actual playbook or house wording — no model changes required.

## One assumption worth checking

The pipeline expects a `Time_Period` feature with categories `Lunch Peak`,
`Dinner Peak`, `Off-Peak`, which isn't a raw column in `orders.csv` — it
must have been engineered from the order hour during training. Since the
original bucketing rule wasn't in the saved files, `app.py` uses:

```
12:00–14:59 → Lunch Peak
19:00–22:59 → Dinner Peak
everything else → Off-Peak
```

This is a reasonable guess, not something recovered from your training
code. If your real definition differs, edit the `get_time_period()`
function near the top of `app.py` — it's used in both the single and batch
prediction paths, so one edit fixes both.

## Compatibility note

The `.pkl` file was saved with an older scikit-learn than may be installed
in your environment (a private attribute on `SimpleImputer` was renamed
between versions, which otherwise raises `AttributeError` on load).
`app.py` includes a small, safe patch in `load_model()` that backfills the
missing attribute so the *exact* trained model keeps working — no
retraining, no behavior change. If you'd rather avoid the patch entirely,
install `scikit-learn==1.7.2` instead and remove the patch call.

## Customizing

A few things you'll likely want to tweak for your own deployment:

- **Currency/labels** — order value, discount, and delivery fee inputs are
  labeled in ₹; change the labels in the "Order details" section of the
  Single Prediction tab if you're using a different currency.
- **Threshold** — currently loaded from `best_threshold.pkl`. To override
  it manually (e.g. for what-if testing), change the fallback value in
  `load_threshold()`.
- **Colors** — the full palette (navy `#1B2A41`, gold `#B08D57`, cream
  `#FAF9F6`) is defined once in the `<style>` block near the top of
  `app.py`; change the hex values there to re-theme the whole app.
- **Recommendation copy** — see [How the recommendations work](#how-the-recommendations-work) above.

## Troubleshooting

**`FileNotFoundError` for the `.pkl` or `.csv` files**
Make sure all seven project files are in the same folder as `app.py`. The
app resolves paths relative to the script's own location, so this works
regardless of which directory you run `streamlit run app.py` from.

**`InconsistentVersionWarning` in the terminal**
Harmless — scikit-learn is warning that the pipeline was fit under a
different version than you have installed. The app already patches the one
known breaking change (see [Compatibility note](#compatibility-note)); the
warning itself doesn't affect predictions.

**Batch upload errors with "Missing required columns"**
Your CSV needs at minimum: `Customer_ID`, `Restaurant_ID`, `Order_Time`,
`Order_Value`, `Discount`, `Delivery_Fee`, `Payment_Method`, `Distance_KM`,
`Rider_Availability`. `Customer_Type`, `Restaurant_Rating`, and
`Avg_Preparation_Time` are joined automatically from `customers.csv` and
`restaurants.csv` by ID.