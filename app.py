"""
Food Delivery Order Cancellation Prediction
--------------------------------------------
A professional Streamlit dashboard for predicting order cancellations
using a pre-trained scikit-learn Logistic Regression pipeline.

IMPORTANT:
- The model is loaded as-is. It is never retrained or modified here.
- All preprocessing (encoding/scaling) lives inside the saved pipeline.
  This app only passes raw feature values into pipeline.predict_proba().
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Order Cancellation Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Classic / professional styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Source Sans 3', sans-serif;
        }

        .main {
            background-color: #FAF9F6;
        }

        h1, h2, h3 {
            font-family: 'Playfair Display', serif;
            color: #1B2A41;
        }

        /* Header banner */
        .header-banner {
            background: linear-gradient(135deg, #1B2A41 0%, #2C3E5C 100%);
            padding: 2rem 2.2rem;
            border-radius: 10px;
            margin-bottom: 1.6rem;
            box-shadow: 0 4px 14px rgba(27, 42, 65, 0.18);
        }
        .header-banner h1 {
            color: #F4EFE6;
            margin: 0;
            font-size: 2.1rem;
        }
        .header-banner p {
            color: #C9D2DE;
            margin: 0.35rem 0 0 0;
            font-size: 0.98rem;
        }

        /* Card containers */
        .info-card {
            background: #FFFFFF;
            border: 1px solid #E7E2D8;
            border-left: 5px solid #B08D57;
            border-radius: 8px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            margin-bottom: 1rem;
        }

        .result-safe {
            background: linear-gradient(135deg, #1E5631 0%, #2E7D46 100%);
            border: none;
            border-left: 6px solid #8FD9A8;
            border-radius: 10px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 4px 16px rgba(30, 86, 49, 0.28);
        }
        .result-risk {
            background: linear-gradient(135deg, #7A241C 0%, #B3382C 100%);
            border: none;
            border-left: 6px solid #F0A79E;
            border-radius: 10px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 4px 16px rgba(122, 36, 28, 0.28);
        }
        .result-safe h2, .result-risk h2 { margin-top: 0; color: #FFFFFF; }
        .result-safe p { color: #E3F5E9; margin-bottom: 0; }
        .result-risk p { color: #FBE3E0; margin-bottom: 0; }
        .result-safe b, .result-risk b { color: #FFFFFF; }

        /* Section dividers */
        .section-label {
            font-family: 'Playfair Display', serif;
            color: #B08D57;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-size: 0.82rem;
            font-weight: 700;
            margin: 0.4rem 0 0.6rem 0;
            border-bottom: 1px solid #E7E2D8;
            padding-bottom: 0.35rem;
        }

        div[data-testid="stMetricValue"] {
            font-family: 'Playfair Display', serif;
            color: #1B2A41;
        }

        .stButton>button {
            background-color: #1B2A41;
            color: #F4EFE6;
            border-radius: 6px;
            border: none;
            padding: 0.55rem 1.6rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .stButton>button:hover {
            background-color: #B08D57;
            color: #1B2A41;
        }

        section[data-testid="stSidebar"] {
            background-color: #1B2A41;
            border-right: 1px solid #10192B;
        }
        section[data-testid="stSidebar"] * {
            color: #E9E4D8;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #F4EFE6;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(233, 228, 216, 0.25);
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] small {
            color: #C9D2DE !important;
        }
        section[data-testid="stSidebar"] .streamlit-expanderHeader,
        section[data-testid="stSidebar"] details summary {
            background-color: rgba(255, 255, 255, 0.06);
            border-radius: 6px;
            color: #F4EFE6 !important;
        }
        section[data-testid="stSidebar"] details {
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(233, 228, 216, 0.15);
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        section[data-testid="stSidebar"] code {
            background-color: rgba(255, 255, 255, 0.1);
            color: #F0D9A8;
        }
        section[data-testid="stSidebar"] .brand-badge {
            background: #B08D57;
            color: #1B2A41;
        }

        /* Recommendation cards */
        .rec-card {
            display: flex;
            gap: 0.7rem;
            align-items: flex-start;
            background: #FFFFFF;
            border: 1px solid #E7E2D8;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        }
        .rec-card .rec-icon { font-size: 1.15rem; margin-top: 0.05rem; }
        .rec-card .rec-title {
            font-weight: 700; color: #1B2A41; font-size: 0.94rem; margin-bottom: 0.15rem;
        }
        .rec-card .rec-text { color: #4A4F5A; font-size: 0.88rem; line-height: 1.4; }
        .rec-card.protective { border-left: 4px solid #2E7D46; }
        .rec-card.risk { border-left: 4px solid #B3382C; }

        .kpi-strip {
            display: flex; gap: 0.9rem; flex-wrap: wrap; margin-bottom: 0.6rem;
        }

        .brand-badge {
            display: inline-block; background: #1B2A41; color: #F4EFE6;
            border-radius: 20px; padding: 0.15rem 0.75rem; font-size: 0.72rem;
            letter-spacing: 0.5px; font-weight: 600; margin-right: 0.4rem;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
import os
APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(APP_DIR, "model/food_delivery_logistic_model.pkl")
THRESHOLD_PATH = os.path.join(APP_DIR, "model/best_threshold.pkl")
CUSTOMERS_PATH = os.path.join(APP_DIR, "Data/customers.csv")
RESTAURANTS_PATH = os.path.join(APP_DIR, "Data/restaurants.csv")
ORDERS_PATH = os.path.join(APP_DIR, "Data/orders.csv")

NUMERIC_FEATURES = [
    "Order_Value", "Discount", "Delivery_Fee", "Distance_KM",
    "Order_Hour", "Avg_Preparation_Time", "Restaurant_Rating",
]
CATEGORICAL_FEATURES = [
    "Payment_Method", "Customer_Type", "Rider_Availability", "Time_Period",
]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PAYMENT_METHODS = ["Card", "Cash on Delivery", "UPI", "Wallet"]
CUSTOMER_TYPES = ["New", "Returning"]
RIDER_AVAILABILITY = ["High", "Medium", "Low"]
TIME_PERIODS = ["Lunch Peak", "Dinner Peak", "Off-Peak"]


def _patch_imputer_version_skew(estimator):
    """Compatibility shim.

    This model was pickled with an older scikit-learn than may be installed
    at runtime. Newer scikit-learn's SimpleImputer.transform() reads an
    attribute named `_fill_dtype`, while imputers fit under older versions
    only stored `_fit_dtype`. This walks the pipeline and backfills the
    missing attribute so the *exact* trained model can still be used
    as-is, with no retraining, on newer scikit-learn installs.
    """
    from sklearn.impute import SimpleImputer

    if isinstance(estimator, SimpleImputer):
        if not hasattr(estimator, "_fill_dtype") and hasattr(estimator, "_fit_dtype"):
            estimator._fill_dtype = estimator._fit_dtype
    for attr in ("transformers_",):
        if hasattr(estimator, attr):
            for _, trans, _ in getattr(estimator, attr):
                _patch_imputer_version_skew(trans)
    if hasattr(estimator, "steps"):
        for _, step in estimator.steps:
            _patch_imputer_version_skew(step)


@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(MODEL_PATH)
    _patch_imputer_version_skew(model)
    return model


@st.cache_resource(show_spinner=False)
def load_threshold():
    try:
        return float(joblib.load(THRESHOLD_PATH))
    except Exception:
        return 0.5


@st.cache_data(show_spinner=False)
def load_reference_data():
    customers = pd.read_csv(CUSTOMERS_PATH)
    restaurants = pd.read_csv(RESTAURANTS_PATH)
    orders = pd.read_csv(ORDERS_PATH)
    return customers, restaurants, orders


def get_time_period(hour: int) -> str:
    """Bucket an hour-of-day into the Time_Period categories the
    pipeline's encoder was trained on. Adjust these boundaries if your
    original feature-engineering used a different definition."""
    if 12 <= hour < 15:
        return "Lunch Peak"
    if 19 <= hour < 23:
        return "Dinner Peak"
    return "Off-Peak"


def build_feature_row(order_value, discount, delivery_fee, distance_km,
                       order_hour, avg_prep_time, restaurant_rating,
                       payment_method, customer_type, rider_availability):
    time_period = get_time_period(order_hour)
    row = {
        "Order_Value": order_value,
        "Discount": discount,
        "Delivery_Fee": delivery_fee,
        "Distance_KM": distance_km,
        "Order_Hour": order_hour,
        "Avg_Preparation_Time": avg_prep_time,
        "Restaurant_Rating": restaurant_rating,
        "Payment_Method": payment_method,
        "Customer_Type": customer_type,
        "Rider_Availability": rider_availability,
        "Time_Period": time_period,
    }
    return pd.DataFrame([row])[FEATURE_ORDER], time_period


def risk_gauge(probability: float, threshold: float) -> go.Figure:
    color = "#B3382C" if probability >= threshold else "#2E7D46"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#1B2A41"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#1B2A41"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#FFFFFF",
                "borderwidth": 1,
                "bordercolor": "#E7E2D8",
                "steps": [
                    {"range": [0, threshold * 100], "color": "#EAF4EC"},
                    {"range": [threshold * 100, 100], "color": "#FBEBE9"},
                ],
                "threshold": {
                    "line": {"color": "#1B2A41", "width": 3},
                    "thickness": 0.9,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Source Sans 3"},
    )
    return fig


# --------------------------------------------------------------------------
# Explainability — turn the logistic regression's own coefficients into
# a per-order breakdown of what pushed the probability up or down, and
# translate the biggest drivers into plain-language operational advice.
# --------------------------------------------------------------------------
def get_feature_contributions(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per underlying (encoded) feature with its transformed
    value and its contribution to the model's log-odds (value * coefficient).
    Positive contribution = pushes cancellation probability up."""
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["model"]
    Xt = pre.transform(features_df)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt)[0]
    names = pre.get_feature_names_out()
    contrib = Xt * clf.coef_[0]
    return pd.DataFrame({"feature": names, "value": Xt, "contribution": contrib})


def _clean_feature_name(name: str) -> str:
    return name.split("__", 1)[-1]


NUMERIC_DRIVER_COPY = {
    "Order_Value": {
        "high": ("Higher-than-average order value", "Send a quick confirmation for this high-ticket order — a cancellation here carries more revenue at risk."),
        "low": ("Lower-than-average order value", "Small-basket orders trend slightly higher on cancellations — an add-on prompt or a minimum-order nudge can help."),
    },
    "Discount": {
        "high": ("Larger-than-usual discount applied", "Re-confirm coupon eligibility and the final payable amount to avoid a price-shock cancellation at delivery."),
        "low": None,
    },
    "Delivery_Fee": {
        "high": ("Delivery fee higher than typical", "Consider a partial fee waiver or a free-delivery nudge for this order to reduce hesitation."),
        "low": None,
    },
    "Distance_KM": {
        "high": ("Long delivery distance", "Confirm the drop address up front and set a realistic ETA before dispatch."),
        "low": None,
    },
    "Avg_Preparation_Time": {
        "high": ("Restaurant runs a long average prep time", "Send a proactive ETA update to the customer so kitchen delay doesn't drive impatience-cancellation."),
        "low": None,
    },
    "Restaurant_Rating": {
        "high": None,
        "low": ("Below-average restaurant rating", "Keep this order on a quality/rider watch-list — lower-rated outlets see more drop-off."),
    },
    "Order_Hour": {"high": None, "low": None},
}

CATEGORICAL_DRIVER_COPY = {
    "Payment_Method_Cash on Delivery": ("Cash-on-Delivery payment", "COD orders cancel more often — consider an OTP confirmation call or a small prepaid incentive."),
    "Customer_Type_New": ("First-time customer", "Send an order-confirmation SMS/WhatsApp — first-time customers are more prone to no-shows."),
    "Rider_Availability_Low": ("Low rider availability in this slot", "Assign a priority rider or route to manual dispatch to avoid a delay-driven cancellation."),
    "Rider_Availability_Medium": ("Moderate rider availability", "Keep a backup rider on standby for this slot as a precaution."),
    "Time_Period_Dinner Peak": ("Placed in the dinner peak window", "Expect kitchen congestion — proactively communicate a buffered ETA."),
    "Time_Period_Lunch Peak": ("Placed in the lunch peak window", "Expect kitchen congestion — proactively communicate a buffered ETA."),
}

PROTECTIVE_COPY = {
    "Customer_Type_Returning": ("Returning, loyal customer", "Track record suggests a reliable delivery — standard workflow is sufficient."),
    "Rider_Availability_High": ("High rider availability", "Riders are readily available for this slot — supports an on-time delivery."),
    "Time_Period_Off-Peak": ("Off-peak order timing", "Kitchen and rider load is light at this hour — lower operational strain."),
}


def build_recommendations(contrib_df: pd.DataFrame, top_n: int = 4, min_abs: float = 0.01):
    """Rank risk-increasing drivers and map the top ones to concrete,
    plain-language recommendations. Returns a list of (label, text)."""
    risk_rows = contrib_df[contrib_df["contribution"] > min_abs].sort_values(
        "contribution", ascending=False
    )
    recs, seen = [], set()
    for _, r in risk_rows.iterrows():
        fname = _clean_feature_name(r["feature"])
        item = None
        if fname in CATEGORICAL_DRIVER_COPY:
            item = CATEGORICAL_DRIVER_COPY[fname]
        else:
            for base, variants in NUMERIC_DRIVER_COPY.items():
                if fname == base:
                    direction = "high" if r["value"] > 0 else "low"
                    item = variants.get(direction)
                    break
        if item and item[0] not in seen:
            recs.append(item)
            seen.add(item[0])
        if len(recs) >= top_n:
            break
    return recs


def build_protective_factors(contrib_df: pd.DataFrame, top_n: int = 3, min_abs: float = 0.01):
    """Mirror of build_recommendations for the factors currently keeping
    the order safe — used on the low-risk verdict."""
    safe_rows = contrib_df[contrib_df["contribution"] < -min_abs].sort_values("contribution")
    facts, seen = [], set()
    for _, r in safe_rows.iterrows():
        fname = _clean_feature_name(r["feature"])
        item = PROTECTIVE_COPY.get(fname)
        if not item and fname in NUMERIC_DRIVER_COPY:
            # a numeric feature pulling risk down isn't mapped to specific
            # copy, so skip rather than guess
            item = None
        if item and item[0] not in seen:
            facts.append(item)
            seen.add(item[0])
        if len(facts) >= top_n:
            break
    return facts


def driver_bar_chart(contrib_df: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """Horizontal bar chart of the top absolute contributors to this
    order's predicted log-odds of cancellation."""
    df = contrib_df.copy()
    df["label"] = df["feature"].apply(_clean_feature_name).str.replace("_", " ")
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top_n)
    df = df.iloc[::-1]
    colors = ["#B3382C" if v > 0 else "#2E7D46" for v in df["contribution"]]
    fig = go.Figure(
        go.Bar(
            x=df["contribution"], y=df["label"], orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="What's driving this prediction",
        height=280,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="← lowers risk   |   raises risk →",
        font={"family": "Source Sans 3", "size": 12},
    )
    return fig


# --------------------------------------------------------------------------
# Load everything
# --------------------------------------------------------------------------
model = load_model()
threshold = load_threshold()
customers_df, restaurants_df, orders_df = load_reference_data()

# --------------------------------------------------------------------------
# Sidebar — branding and model documentation
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom:0.4rem;">
            <span class="brand-badge">MODEL v1</span>
            <span class="brand-badge" style="background:rgba(255,255,255,0.12);color:#F4EFE6;border:1px solid rgba(244,239,230,0.4);">LIVE SCORING</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🍽️ Cancellation Intelligence")
    st.caption("Decision-support tool for the operations desk — not a replacement for judgment on edge cases.")

    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown(
            f"""
            - **Algorithm:** Logistic Regression (scikit-learn pipeline)
            - **Status:** loaded and served exactly as trained — no retraining happens in this app
            - **Decision threshold:** `{threshold:.2f}` — orders scoring at or above this are flagged high-risk
            - **Inputs used:**
                - Order value, discount, delivery fee
                - Delivery distance
                - Order hour → time-of-day bucket
                - Restaurant rating & average prep time
                - Payment method
                - Customer type (new / returning)
                - Rider availability
            """
        )

    with st.expander("⚠️ Known assumption", expanded=False):
        st.markdown(
            """
            - The model expects a `Time_Period` bucket that isn't a raw
              column in the order data — it's **derived here** from the order hour:
                - `12:00–14:59` → **Lunch Peak**
                - `19:00–22:59` → **Dinner Peak**
                - everything else → **Off-Peak**
            - If your original feature engineering used different cut-offs,
              update the `get_time_period()` function in `app.py` to match.
            """
        )

    st.markdown("---")
    st.markdown("**📦 Reference data loaded**")
    st.markdown(
        f"""
        - 👥 {len(customers_df):,} customers
        - 🏬 {len(restaurants_df):,} restaurants
        - 🧾 {len(orders_df):,} historical orders
        """
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-banner">
        <h1>🍽️ Order Cancellation Risk Predictor</h1>
        <p>Score orders in real time, see exactly what's driving the risk, and get
        concrete next-best-actions for the operations desk — powered by a
        pre-trained, frozen Logistic Regression pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_batch, tab_insights = st.tabs(
    ["🔮 Single Prediction", "📄 Batch Prediction", "📊 Portfolio Insights"]
)

# ==========================================================================
# TAB 1 — Single order prediction
# ==========================================================================
with tab_predict:
    left, right = st.columns([1.05, 1.4], gap="large")

    with left:
        st.markdown('<div class="section-label">Customer</div>', unsafe_allow_html=True)
        customer_mode = st.radio(
            "Customer source", ["Lookup existing", "Manual entry"],
            horizontal=True, label_visibility="collapsed",
        )
        if customer_mode == "Lookup existing":
            cust_id = st.selectbox("Customer ID", customers_df["Customer_ID"].sort_values())
            cust_row = customers_df.loc[customers_df["Customer_ID"] == cust_id].iloc[0]
            customer_type = cust_row["Customer_Type"]
            st.caption(
                f"**Type:** {cust_row['Customer_Type']}  |  **Age group:** {cust_row['Age_Group']}  |  "
                f"**Area:** {cust_row['Area']}  |  **Past orders:** {int(cust_row['Total_Previous_Orders'])}"
            )
        else:
            customer_type = st.selectbox("Customer type", CUSTOMER_TYPES)

        st.markdown('<div class="section-label">Restaurant</div>', unsafe_allow_html=True)
        restaurant_mode = st.radio(
            "Restaurant source", ["Lookup existing", "Manual entry"],
            horizontal=True, label_visibility="collapsed", key="rest_mode",
        )
        if restaurant_mode == "Lookup existing":
            rest_label = st.selectbox(
                "Restaurant",
                restaurants_df.apply(lambda r: f"{r['Restaurant_Name']} ({r['Restaurant_ID']})", axis=1),
            )
            rest_id = rest_label.split("(")[-1].rstrip(")")
            rest_row = restaurants_df.loc[restaurants_df["Restaurant_ID"] == rest_id].iloc[0]
            restaurant_rating = float(rest_row["Restaurant_Rating"])
            avg_prep_time = int(rest_row["Avg_Preparation_Time"])
            st.caption(
                f"**Cuisine:** {rest_row['Cuisine']}  |  **Area:** {rest_row['Area']}  |  "
                f"**Rating:** {restaurant_rating}  |  **Avg prep time:** {avg_prep_time} min"
            )
        else:
            restaurant_rating = st.slider("Restaurant rating", 3.0, 5.0, 4.2, 0.1)
            avg_prep_time = st.slider("Avg preparation time (min)", 10, 60, 25)

        st.markdown('<div class="section-label">Order details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            order_value = st.number_input("Order value (₹)", min_value=0.0, value=350.0, step=10.0)
            delivery_fee = st.number_input("Delivery fee (₹)", min_value=0.0, value=45.0, step=5.0)
            distance_km = st.slider("Distance (km)", 0.5, 20.0, 4.0, 0.1)
        with c2:
            discount = st.number_input("Discount (₹)", min_value=0.0, value=20.0, step=5.0)
            payment_method = st.selectbox("Payment method", PAYMENT_METHODS)
            rider_availability = st.selectbox("Rider availability", RIDER_AVAILABILITY)

        order_hour = st.slider("Order hour (24h)", 0, 23, 20)
        st.caption(f"Derived time bucket → **{get_time_period(order_hour)}**")

        predict_clicked = st.button("Predict cancellation risk", use_container_width=True)

    with right:
        if predict_clicked:
            features_df, time_period = build_feature_row(
                order_value, discount, delivery_fee, distance_km, order_hour,
                avg_prep_time, restaurant_rating, payment_method,
                customer_type, rider_availability,
            )
            proba = float(model.predict_proba(features_df)[0, 1])
            will_cancel = proba >= threshold

            st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(proba, threshold), use_container_width=True)

            contrib_df = get_feature_contributions(model, features_df)

            if will_cancel:
                st.markdown(
                    f"""<div class="result-risk">
                        <h2>⚠️ High Cancellation Risk</h2>
                        <p>Predicted probability of cancellation is <b>{proba:.1%}</b>,
                        above the decision threshold of <b>{threshold:.1%}</b>.
                        Proactive intervention before dispatch is recommended —
                        see the suggested actions below.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="result-safe">
                        <h2>✅ Low Cancellation Risk</h2>
                        <p>Predicted probability of cancellation is <b>{proba:.1%}</b>,
                        below the decision threshold of <b>{threshold:.1%}</b>.
                        This order can proceed through the standard flow.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-label">What\'s driving this score</div>', unsafe_allow_html=True)
            st.plotly_chart(driver_bar_chart(contrib_df), use_container_width=True)

            st.markdown('<div class="section-label">Recommended actions</div>', unsafe_allow_html=True)
            recommendations = build_recommendations(contrib_df) if will_cancel else []
            if recommendations:
                for label, text in recommendations:
                    st.markdown(
                        f"""<div class="rec-card risk">
                            <div class="rec-icon">🎯</div>
                            <div>
                                <div class="rec-title">{label}</div>
                                <div class="rec-text">{text}</div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            elif will_cancel:
                st.markdown(
                    """<div class="rec-card risk">
                        <div class="rec-icon">🎯</div>
                        <div>
                            <div class="rec-title">No single dominant driver</div>
                            <div class="rec-text">Risk is spread across several moderate factors rather than one
                            clear cause — a standard confirmation call before dispatch is still advisable.</div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                protective = build_protective_factors(contrib_df)
                if protective:
                    for label, text in protective:
                        st.markdown(
                            f"""<div class="rec-card protective">
                                <div class="rec-icon">✔️</div>
                                <div>
                                    <div class="rec-title">{label}</div>
                                    <div class="rec-text">{text}</div>
                                </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        """<div class="rec-card protective">
                            <div class="rec-icon">✔️</div>
                            <div>
                                <div class="rec-title">No action needed</div>
                                <div class="rec-text">No significant risk drivers detected — proceed through the standard workflow.</div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            with st.expander("Show feature values sent to the model"):
                st.dataframe(features_df.T.rename(columns={0: "Value"}), use_container_width=True)
        else:
            st.markdown(
                """<div class="info-card">
                    <b>How this works</b><br>
                    Fill in the customer, restaurant and order details on the left,
                    then click <i>Predict cancellation risk</i>. You'll get a
                    probability gauge, a breakdown of exactly which factors are
                    pushing the risk up or down, and concrete next-best-actions
                    for the operations desk — all from the saved Logistic
                    Regression pipeline, used exactly as trained.
                </div>""",
                unsafe_allow_html=True,
            )

# ==========================================================================
# TAB 2 — Batch prediction
# ==========================================================================
with tab_batch:
    st.markdown('<div class="section-label">Upload orders for bulk scoring</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="info-card">
            Upload a CSV with columns similar to <code>orders.csv</code>:
            <code>Customer_ID, Restaurant_ID, Order_Time, Order_Value, Discount,
            Delivery_Fee, Payment_Method, Distance_KM, Rider_Availability</code>.
            Customer and restaurant attributes are joined automatically from the
            reference tables loaded with this app.
        </div>""",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            merged = batch_df.merge(
                customers_df[["Customer_ID", "Customer_Type"]], on="Customer_ID", how="left"
            ).merge(
                restaurants_df[["Restaurant_ID", "Restaurant_Rating", "Avg_Preparation_Time"]],
                on="Restaurant_ID", how="left",
            )
            merged["Order_Hour"] = pd.to_datetime(
                merged["Order_Time"], format="%H:%M", errors="coerce"
            ).dt.hour
            merged["Time_Period"] = merged["Order_Hour"].apply(
                lambda h: get_time_period(int(h)) if pd.notna(h) else "Off-Peak"
            )

            missing = [c for c in FEATURE_ORDER if c not in merged.columns]
            if missing:
                st.error(f"Missing required columns after merge: {missing}")
            else:
                X = merged[FEATURE_ORDER]
                probs = model.predict_proba(X)[:, 1]
                merged["Cancellation_Probability"] = probs
                merged["Predicted_Cancel"] = (probs >= threshold).map({True: "Yes", False: "No"})

                st.success(f"Scored {len(merged):,} orders.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Orders scored", f"{len(merged):,}")
                m2.metric("Flagged high-risk", f"{(merged['Predicted_Cancel'] == 'Yes').sum():,}")
                m3.metric("Avg. cancellation probability", f"{probs.mean():.1%}")

                st.dataframe(
                    merged[["Order_ID"] if "Order_ID" in merged.columns else []
                           + ["Customer_ID", "Restaurant_ID", "Cancellation_Probability", "Predicted_Cancel"]],
                    use_container_width=True,
                )

                fig = px.histogram(
                    merged, x="Cancellation_Probability", nbins=30,
                    color_discrete_sequence=["#1B2A41"],
                    title="Distribution of predicted cancellation probability",
                )
                fig.add_vline(x=threshold, line_dash="dash", line_color="#B08D57")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-label">Top risk drivers across flagged orders</div>', unsafe_allow_html=True)
                flagged = merged[merged["Predicted_Cancel"] == "Yes"]
                if len(flagged) > 0:
                    sample = flagged.sample(min(len(flagged), 500), random_state=42)
                    driver_counts = {}
                    for _, order_row in sample[FEATURE_ORDER].iterrows():
                        cdf = get_feature_contributions(model, pd.DataFrame([order_row]))
                        for label, _ in build_recommendations(cdf, top_n=2):
                            driver_counts[label] = driver_counts.get(label, 0) + 1
                    if driver_counts:
                        summary = pd.DataFrame(
                            sorted(driver_counts.items(), key=lambda x: -x[1]),
                            columns=["Driver", "Orders affected"],
                        )
                        summary["Share of flagged orders"] = summary["Orders affected"] / len(sample)
                        figd = px.bar(
                            summary.head(8), x="Share of flagged orders", y="Driver", orientation="h",
                            color_discrete_sequence=["#B3382C"],
                        )
                        figd.update_layout(
                            xaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=10, t=20, b=10), height=280,
                        )
                        st.plotly_chart(figd, use_container_width=True)
                        st.caption("Based on a sample of up to 500 flagged orders. Use this to prioritize which operational fix (rider allocation, COD confirmation, peak-hour buffering, etc.) will move the needle most.")
                    else:
                        st.caption("No single dominant driver found across the flagged orders — risk appears spread across many small factors.")
                else:
                    st.caption("No orders were flagged as high-risk in this batch.")

                csv_out = merged.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scored results", csv_out,
                    file_name="scored_orders.csv", mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ==========================================================================
# TAB 3 — Portfolio insights (from historical orders.csv)
# ==========================================================================
with tab_insights:
    st.markdown('<div class="section-label">Historical order book overview</div>', unsafe_allow_html=True)

    total_orders = len(orders_df)
    cancelled = (orders_df["Order_Status"] == "Cancelled").sum()
    cancel_rate = cancelled / total_orders if total_orders else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total orders", f"{total_orders:,}")
    k2.metric("Cancelled orders", f"{cancelled:,}")
    k3.metric("Historical cancellation rate", f"{cancel_rate:.1%}")
    k4.metric("Current decision threshold", f"{threshold:.1%}")

    colA, colB = st.columns(2)
    palette = ["#1B2A41", "#B08D57", "#5D7A8C", "#B3382C", "#7A8B99"]

    with colA:
        by_payment = (
            orders_df.assign(Cancelled=orders_df["Order_Status"].eq("Cancelled"))
            .groupby("Payment_Method")["Cancelled"].mean().reset_index()
        )
        fig1 = px.bar(
            by_payment, x="Payment_Method", y="Cancelled",
            title="Cancellation rate by payment method",
            color_discrete_sequence=[palette[0]],
        )
        fig1.update_layout(
            yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig1, use_container_width=True)

        by_rider = (
            orders_df.assign(Cancelled=orders_df["Order_Status"].eq("Cancelled"))
            .groupby("Rider_Availability")["Cancelled"].mean().reset_index()
        )
        fig3 = px.bar(
            by_rider, x="Rider_Availability", y="Cancelled",
            title="Cancellation rate by rider availability",
            color_discrete_sequence=[palette[1]],
            category_orders={"Rider_Availability": ["High", "Medium", "Low"]},
        )
        fig3.update_layout(
            yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with colB:
        order_hour = pd.to_datetime(orders_df["Order_Time"], format="%H:%M", errors="coerce").dt.hour
        by_hour = (
            orders_df.assign(Cancelled=orders_df["Order_Status"].eq("Cancelled"), Hour=order_hour)
            .groupby("Hour")["Cancelled"].mean().reset_index()
        )
        fig2 = px.line(
            by_hour, x="Hour", y="Cancelled", markers=True,
            title="Cancellation rate by hour of day",
            color_discrete_sequence=[palette[3]],
        )
        fig2.update_layout(
            yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        dist_bucket = pd.cut(
            orders_df["Distance_KM"], bins=[0, 3, 6, 10, 20],
            labels=["0-3 km", "3-6 km", "6-10 km", "10-20 km"],
        )
        by_dist = (
            orders_df.assign(Cancelled=orders_df["Order_Status"].eq("Cancelled"), Bucket=dist_bucket)
            .groupby("Bucket", observed=True)["Cancelled"].mean().reset_index()
        )
        fig4 = px.bar(
            by_dist, x="Bucket", y="Cancelled",
            title="Cancellation rate by distance",
            color_discrete_sequence=[palette[2]],
        )
        fig4.update_layout(
            yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-label">Where cancellations concentrate</div>', unsafe_allow_html=True)
    orders_enriched = orders_df.merge(
        restaurants_df[["Restaurant_ID", "Area", "Cuisine"]], on="Restaurant_ID", how="left"
    ).assign(Cancelled=lambda d: d["Order_Status"].eq("Cancelled"))

    colC, colD = st.columns(2)
    with colC:
        by_area = (
            orders_enriched.groupby("Area", observed=True)
            .agg(Cancel_Rate=("Cancelled", "mean"), Orders=("Cancelled", "size"))
            .reset_index().sort_values("Cancel_Rate", ascending=False)
        )
        fig5 = px.bar(
            by_area, x="Cancel_Rate", y="Area", orientation="h",
            title="Cancellation rate by restaurant area",
            color_discrete_sequence=[palette[0]], hover_data=["Orders"],
        )
        fig5.update_layout(
            xaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig5, use_container_width=True)

    with colD:
        by_cuisine = (
            orders_enriched.groupby("Cuisine", observed=True)
            .agg(Cancel_Rate=("Cancelled", "mean"), Orders=("Cancelled", "size"))
            .reset_index().sort_values("Cancel_Rate", ascending=False)
        )
        fig6 = px.bar(
            by_cuisine, x="Cancel_Rate", y="Cuisine", orientation="h",
            title="Cancellation rate by cuisine",
            color_discrete_sequence=[palette[3]], hover_data=["Orders"],
        )
        fig6.update_layout(
            xaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig6, use_container_width=True)

    top_area = by_area.iloc[0]
    top_cuisine = by_cuisine.iloc[0]
    st.markdown(
        f"""<div class="info-card">
            <b>Quick read:</b> <code>{top_area['Area']}</code> has the highest
            cancellation rate among restaurant areas at <b>{top_area['Cancel_Rate']:.1%}</b>
            ({int(top_area['Orders'])} orders), and <code>{top_cuisine['Cuisine']}</code>
            leads among cuisines at <b>{top_cuisine['Cancel_Rate']:.1%}</b>
            ({int(top_cuisine['Orders'])} orders). Worth a closer look if either
            has enough volume to be statistically meaningful.
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    """<hr style="border-color:#E7E2D8; margin-top:2rem;">
    <p style="text-align:center; color:#8B93A0; font-size:0.82rem;">
    Order Cancellation Risk Predictor &middot; model served as-is, no retraining performed here
    </p>""",
    unsafe_allow_html=True,
)