import io
import os

import numpy as np
import pandas as pd
import streamlit as st

import churn_ml as cm

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle():
    if not os.path.exists(cm.MODEL_PATH):
        with st.spinner("First run: training the model (about a minute)..."):
            cm.train(verbose=False)
    return cm.load_bundle()


bundle = get_bundle()
model, baselines, background = bundle["model"], bundle["baselines"], bundle["background"]

st.title("Customer Churn Predictor")
st.caption("Predict which customers are about to leave, understand why, "
           "and decide who is worth a retention offer.")

tab_score, tab_impact, tab_whatif, tab_report = st.tabs(
    ["1. Score customers", "2. Business impact", "3. What-if (single customer)", "4. Model report"])


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def score(df_bytes):
    raw = pd.read_csv(io.BytesIO(df_bytes))
    X = cm.prepare(raw)
    p, deltas = cm.explain(model, X, background)
    out = raw.copy()
    out["churn_probability"] = p.round(3)
    out["risk"] = pd.cut(p, [-1, 0.2, 0.5, 2], labels=["Low", "Medium", "High"])
    out["why_at_risk"] = [cm.top_reasons(X.loc[i], deltas.loc[i]) for i in X.index]
    return out, X, deltas


# ------------------------------------------------------------------ tab 1
with tab_score:
    st.write("Upload a customer CSV (same columns as the Telco dataset) or use the built-in sample.")
    up = st.file_uploader("Customer CSV", type="csv")
    use_sample = st.button("Use sample customers (300 held-out customers)")
    if use_sample:
        st.session_state["csv_bytes"] = open(cm.SAMPLE_PATH, "rb").read()
    if up is not None:
        st.session_state["csv_bytes"] = up.getvalue()

    if "csv_bytes" in st.session_state:
        try:
            scored, X, deltas = score(st.session_state["csv_bytes"])
        except Exception as e:
            st.error(f"Could not score this file: {e}")
            st.stop()
        st.session_state["scored"] = (scored, X)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", len(scored))
        c2.metric("High risk", int((scored["risk"] == "High").sum()))
        c3.metric("Average churn risk", f"{scored['churn_probability'].mean():.0%}")
        c4.metric("Expected churners", f"{scored['churn_probability'].sum():.0f}")

        view = scored.sample(frac=1).reset_index(drop=True)
        cols = [c for c in ["customerID", "churn_probability", "risk", "why_at_risk", "Contract", "tenure",
                            "MonthlyCharges"] if c in view.columns]
        st.dataframe(view[cols], width="stretch", hide_index=True,
                     column_config={"churn_probability": st.column_config.ProgressColumn(
                         "churn probability", min_value=0.0, max_value=1.0, format="%.2f")})
        st.download_button("Download scored customers (CSV)", view.to_csv(index=False),
                           "scored_customers.csv", "text/csv")

        st.subheader("What drives churn in this customer base?")
        avg_push = deltas.clip(lower=0).mean().sort_values(ascending=False).head(8) * 100
        st.bar_chart(avg_push.rename("avg. risk added (percentage points)"))
    else:
        st.info("Click the sample button to see it in action.")


# ------------------------------------------------------------------ tab 2
with tab_impact:
    st.write("Turn predictions into a **decision**: who should get a retention offer so the "
             "campaign actually makes money?")
    if "scored" not in st.session_state:
        st.info("Score some customers in tab 1 first.")
    else:
        scored, _ = st.session_state["scored"]
        a, b, c = st.columns(3)
        offer_cost = a.number_input("Cost of one retention offer ($)", 0.0, 500.0, 20.0, 1.0)
        success = b.slider("Offer success rate", 0.05, 0.9, 0.30, 0.05,
                           help="Share of at-risk customers an offer actually keeps")
        months = c.slider("Months of revenue saved per kept customer", 1, 36, 12)

        gain = cm.retention_value(scored["churn_probability"].values, scored["MonthlyCharges"].values,
                                  months, offer_cost, success)
        order = np.argsort(-gain)
        target = gain > 0
        cum = np.cumsum(gain[order])
        best_k = int(np.argmax(cum)) + 1 if cum.max() > 0 else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Customers worth targeting", int(target.sum()))
        m2.metric("Expected net benefit (model-based)", f"${gain[target].sum():,.0f}")
        m3.metric("If you targeted EVERYONE", f"${gain.sum():,.0f}",
                  delta=f"{gain[target].sum() - gain.sum():,.0f} better with the model")

        st.line_chart(pd.DataFrame({"expected cumulative net benefit ($)": cum},
                                   index=pd.RangeIndex(1, len(cum) + 1, name="customers targeted (highest value first)")))
        st.caption(f"Best strategy: target the top {best_k} customers. "
                   "Figures are expected values based on the assumptions above, not guarantees.")


# ------------------------------------------------------------------ tab 3
with tab_whatif:
    st.write("Change a customer's details and watch the churn risk update. "
             "Try switching **Contract** from month-to-month to one year.")
    values, cols = {}, st.columns(4)
    for i, feat in enumerate(cm.RAW_FEATURES):
        with cols[i % 4]:
            if feat in cm.RAW_NUMERIC:
                lo, hi = bundle["ranges"][feat]
                base = float(baselines[feat])
                values[feat] = st.number_input(feat, lo, hi, base, 1.0 if feat == "tenure" else 5.0,
                                               key=f"w_{feat}")
            else:
                opts = bundle["choices"][feat]
                values[feat] = st.selectbox(feat, opts, index=opts.index(baselines[feat]), key=f"w_{feat}")
    row = pd.DataFrame([values])
    Xw = cm.prepare(row)
    p_w, d_w = cm.explain(model, Xw, background)
    p_w, d_w = float(p_w[0]), d_w.iloc[0]

    left, right = st.columns([1, 2])
    left.metric("Churn probability", f"{p_w:.0%}")
    left.progress(min(max(p_w, 0.0), 1.0))
    left.caption("Calibrated: of customers scored like this, about this share actually leave.")
    drivers = (d_w * 100).sort_values()
    drivers = pd.concat([drivers.head(4), drivers.tail(6)]).drop_duplicates()
    right.write("**What pushes this customer's risk up (+) or down (−), vs. a typical customer (pts):**")
    right.bar_chart(drivers.rename("risk change (percentage points)"))


# ------------------------------------------------------------------ tab 4
with tab_report:
    m = bundle["metrics"]
    st.write(f"Trained on **{m['n_train']:,}** customers, tested on **{m['n_test']:,}** the model never saw. "
             f"Overall churn rate: **{m['churn_rate']:.0%}**.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winning model", m["best_model"])
    c2.metric("ROC-AUC (test)", f"{m['test_roc_auc']:.3f}")
    c3.metric("PR-AUC (test)", f"{m['test_pr_auc']:.3f}")
    c4.metric("Top 20% captures", f"{m['top20_capture']:.0%} of churners",
              delta=f"{m['top20_lift']:.1f}x better than random")

    st.subheader("Model comparison (5-fold cross-validation)")
    st.dataframe(pd.DataFrame(bundle["comparison"]).round(3), width="stretch", hide_index=True)
    if os.path.exists(cm.REPORT_PATH):
        st.image(cm.REPORT_PATH, width="stretch")
    st.markdown("""
**How to read this**
- **ROC-AUC**: chance that a random churner is ranked above a random non-churner (0.5 = coin flip, 1.0 = perfect).
- **Calibration**: when the model says 30%, about 30% really churn, so probabilities can be used for money decisions.
- **Top 20% captures**: if you can only contact one in five customers, this is the share of leavers you reach.
""")
