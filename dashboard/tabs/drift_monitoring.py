# drift_monitoring.py
# Drift Monitoring tab — compares the distribution of incoming synthetic orders
# against the training data baseline to detect feature drift
# Uses Population Stability Index (PSI) as the drift metric

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# Training baseline distributions derived from the features_engineered Delta table
# Numeric features store bin proportions, binary features store the positive class rate
BASELINE_DISTRIBUTIONS = {
    "days_scheduled": {
        0: 0.02, 1: 0.12, 2: 0.18, 3: 0.22, 4: 0.20, 5: 0.16, 6: 0.10
    },
    "suspected_fraud_rate":     0.0225,  # 2.25% positive rate in training data
    "same_day_rate":            0.07,    # ~7% of orders used Same Day shipping
    "aggressive_discount_rate": 0.111    # ~11.1% of orders had aggressive discount tier
}


def compute_psi_continuous(baseline_dist: dict, current_dist: dict) -> float:
    # PSI for continuous features — compares bin proportions across all buckets
    # baseline_dist and current_dist are both value-to-proportion dicts
    all_keys = sorted(set(list(baseline_dist.keys()) + list(current_dist.keys())))
    psi = 0.0
    for k in all_keys:
        expected = max(baseline_dist.get(k, 0.0), 1e-4)
        actual   = max(current_dist.get(k, 0.0), 1e-4)
        psi     += (actual - expected) * np.log(actual / expected)
    return round(psi, 4)


def compute_psi_binary(baseline_rate: float, current_rate: float) -> float:
    # PSI for binary features — compares positive and negative class proportions directly
    # Histogram binning fails for low-prevalence binary features so proportion comparison is used
    p_base = max(baseline_rate, 1e-4)
    p_prod = max(current_rate, 1e-4)
    q_base = max(1 - baseline_rate, 1e-4)
    q_prod = max(1 - current_rate, 1e-4)
    psi    = (p_prod - p_base) * np.log(p_prod / p_base) + \
             (q_prod - q_base) * np.log(q_prod / q_base)
    return round(psi, 4)


def get_psi_status(psi: float) -> tuple:
    # Returns a display label and color string based on PSI thresholds
    if psi < 0.10:
        return "✅ Stable", "green"
    elif psi < 0.20:
        return "⚠️ Moderate Shift", "orange"
    else:
        return "🚨 Significant Shift", "red"


def derive_current_distributions(df: pd.DataFrame) -> dict:
    # Derives current production distributions from the scored risk queue
    # Days Scheduled is already numeric — compute value proportions directly
    days_counts = df["Days Scheduled"].value_counts(normalize=True).to_dict()
    days_dist   = {k: round(v, 4) for k, v in days_counts.items()}

    # Binary features derived from string columns in the risk queue
    suspected_fraud_rate     = (df["Order Status"] == "Suspected Fraud").mean()
    same_day_rate            = (df["Shipping Mode"] == "Same Day").mean()

    # Aggressive discount tier — orders with discount rate above 0.25 match training logic
    # order_item_discount_rate is stored in raw_orders, not risk_queue
    # Pull from raw_orders using session state
    raw_df                   = pd.DataFrame(st.session_state.raw_orders)
    aggressive_discount_rate = (raw_df["order_item_discount_rate"] > 0.25).mean()

    return {
        "days_dist":              days_dist,
        "suspected_fraud_rate":   round(suspected_fraud_rate, 4),
        "same_day_rate":          round(same_day_rate, 4),
        "aggressive_discount_rate": round(aggressive_discount_rate, 4)
    }


def render_drift_monitoring():
    st.subheader("📡 Drift Monitoring")
    st.write(
        "Compares the distribution of scored orders against the training data baseline. "
        "Drift indicates the model may be seeing order patterns it was not trained on."
    )

    if "risk_queue" not in st.session_state or len(st.session_state.risk_queue) < 10:
        st.info(
            "Not enough scored orders to analyze drift. "
            "Go to the Risk Queue tab and generate at least 10 orders first."
        )
        return

    df  = pd.DataFrame(st.session_state.risk_queue)
    current = derive_current_distributions(df)

    st.markdown(f"**Analyzing {len(df)} scored orders against training baseline**")
    st.divider()

    # PSI calculations
    psi_days     = compute_psi_continuous(
        BASELINE_DISTRIBUTIONS["days_scheduled"],
        current["days_dist"]
    )
    psi_fraud    = compute_psi_binary(
        BASELINE_DISTRIBUTIONS["suspected_fraud_rate"],
        current["suspected_fraud_rate"]
    )
    psi_same_day = compute_psi_binary(
        BASELINE_DISTRIBUTIONS["same_day_rate"],
        current["same_day_rate"]
    )
    psi_discount = compute_psi_binary(
        BASELINE_DISTRIBUTIONS["aggressive_discount_rate"],
        current["aggressive_discount_rate"]
    )

    psi_scores = {
        "Days Scheduled":        psi_days,
        "Suspected Fraud Rate":  psi_fraud,
        "Same Day Shipping Rate": psi_same_day,
        "Aggressive Discount Rate": psi_discount
    }

    # PSI summary metric cards
    st.markdown("**Population Stability Index (PSI) Summary**")
    st.caption("PSI < 0.10: stable | 0.10 to 0.20: moderate shift | > 0.20: significant shift")

    cols = st.columns(4)
    for i, (label, psi) in enumerate(psi_scores.items()):
        status_label, _ = get_psi_status(psi)
        with cols[i]:
            st.metric(
                label=label,
                value=f"PSI: {psi:.4f}",
                delta=status_label,
                delta_color="off"
            )

    st.divider()

    # Distribution comparison charts
    st.markdown("**Distribution Comparison — Baseline vs Incoming Orders**")

    # Days Scheduled — grouped bar chart across all bin values
    st.markdown("**Days Scheduled for Shipment**")
    all_days  = sorted(set(list(BASELINE_DISTRIBUTIONS["days_scheduled"].keys()) +
                           list(current["days_dist"].keys())))
    base_vals = [BASELINE_DISTRIBUTIONS["days_scheduled"].get(d, 0) for d in all_days]
    curr_vals = [current["days_dist"].get(d, 0) for d in all_days]

    fig_days = go.Figure()
    fig_days.add_trace(go.Bar(
        name="Training Baseline", x=[str(d) for d in all_days],
        y=base_vals, marker_color="#4a90d9", opacity=0.8
    ))
    fig_days.add_trace(go.Bar(
        name="Incoming Orders", x=[str(d) for d in all_days],
        y=curr_vals, marker_color="#e8513a", opacity=0.8
    ))
    fig_days.update_layout(
        barmode="group", height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_tickformat=".0%"
    )
    st.plotly_chart(fig_days, use_container_width=True)
    
    st.caption(
        "Note: PSI for continuous features is sensitive to small sample sizes. "
        "Results stabilize with 50 or more scored orders in the queue."
    )

    # Binary features — two-bar chart showing positive class rate only
    binary_features = {
        "Suspected Fraud Rate":     (BASELINE_DISTRIBUTIONS["suspected_fraud_rate"],
                                     current["suspected_fraud_rate"]),
        "Same Day Shipping Rate":   (BASELINE_DISTRIBUTIONS["same_day_rate"],
                                     current["same_day_rate"]),
        "Aggressive Discount Rate": (BASELINE_DISTRIBUTIONS["aggressive_discount_rate"],
                                     current["aggressive_discount_rate"])
    }

    for label, (base_rate, curr_rate) in binary_features.items():
        st.markdown(f"**{label}**")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Training Baseline", x=["Positive Rate"],
            y=[base_rate], marker_color="#4a90d9", opacity=0.8
        ))
        fig.add_trace(go.Bar(
            name="Incoming Orders", x=["Positive Rate"],
            y=[curr_rate], marker_color="#e8513a", opacity=0.8
        ))
        fig.update_layout(
            barmode="group", height=250,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_tickformat=".1%"
        )
        st.plotly_chart(fig, use_container_width=True)
        

    st.divider()

    st.markdown("**What to do if drift is detected**")
    st.markdown("""
    - **Moderate shift** — monitor closely, no immediate action required
    - **Significant shift** — investigate whether order patterns have genuinely changed
    - If drift persists across multiple scoring sessions, retrain the model on more recent data and redeploy
    """)