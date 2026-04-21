# drift_monitoring.py
# Drift Monitoring tab — compares the distribution of incoming synthetic orders
# against the training data baseline to detect feature drift
# Uses Population Stability Index (PSI) as the drift metric

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# Training data baseline distributions derived from the features_engineered Delta table
# These represent the expected distribution of each feature at training time
BASELINE_DISTRIBUTIONS = {
    "days_for_shipment_scheduled": {1: 0.12, 2: 0.18, 3: 0.22, 4: 0.20, 5: 0.16, 6: 0.12},
    "shipping_mode": {
        "Standard Class": 0.59,
        "Second Class": 0.19,
        "First Class": 0.15,
        "Same Day": 0.07
    },
    "order_status": {
        "COMPLETE": 0.47,
        "PENDING": 0.17,
        "PROCESSING": 0.14,
        "CLOSED": 0.10,
        "ON_HOLD": 0.07,
        "SUSPECTED_FRAUD": 0.05
    },
    "market": {
        "USCA": 0.31,
        "Europe": 0.28,
        "LATAM": 0.22,
        "Pacific Asia": 0.13,
        "Africa": 0.06
    }
}

# PSI thresholds — standard industry interpretation
PSI_THRESHOLDS = {
    "No drift": (0.0, 0.10),
    "Moderate drift": (0.10, 0.20),
    "Significant drift": (0.20, float("inf"))
}


def compute_psi(baseline: dict, current: dict) -> float:
    # Population Stability Index measures how much a distribution has shifted
    # PSI = sum((current% - baseline%) * ln(current% / baseline%))
    psi = 0.0
    for key in baseline:
        expected = baseline.get(key, 0.001)
        actual = current.get(key, 0.001)
        # Clip to avoid log(0)
        expected = max(expected, 0.001)
        actual = max(actual, 0.001)
        psi += (actual - expected) * np.log(actual / expected)
    return round(psi, 4)


def get_psi_status(psi: float) -> tuple:
    if psi < 0.10:
        return "✅ No drift", "green"
    elif psi < 0.20:
        return "⚠️ Moderate drift", "orange"
    else:
        return "🚨 Significant drift", "red"


def compute_current_distribution(df: pd.DataFrame, feature: str) -> dict:
    # Compute the frequency distribution of a feature from the scored orders
    counts = df[feature].value_counts(normalize=True).to_dict()
    return counts


def render_drift_monitoring():
    st.subheader("📡 Drift Monitoring")
    st.write(
        "Compares the distribution of scored orders against the training data baseline. "
        "Drift indicates the model may be seeing order patterns it was not trained on."
    )

    # Check if there are scored orders in the risk queue to analyze
    if "risk_queue" not in st.session_state or len(st.session_state.risk_queue) < 10:
        st.info(
            "Not enough scored orders to analyze drift. "
            "Go to the Risk Queue tab and generate at least 10 orders first."
        )
        return

    df = pd.DataFrame(st.session_state.risk_queue)

    st.markdown(f"**Analyzing {len(df)} scored orders against training baseline**")
    st.divider()

    # PSI summary cards
    st.markdown("**Population Stability Index (PSI) Summary**")
    st.caption("PSI < 0.10: stable | PSI 0.10 to 0.20: moderate drift | PSI > 0.20: significant drift")

    features_to_monitor = {
        "shipping_mode": "shipping_mode",
        "order_status": "order_status",
        "market": "market"
    }

    psi_results = {}
    cols = st.columns(len(features_to_monitor))

    for i, (feature, col_name) in enumerate(features_to_monitor.items()):
        if col_name in df.columns:
            current_dist = compute_current_distribution(df, col_name)
            psi = compute_psi(BASELINE_DISTRIBUTIONS[feature], current_dist)
            status_label, status_color = get_psi_status(psi)
            psi_results[feature] = {"psi": psi, "status": status_label, "current": current_dist}

            with cols[i]:
                st.metric(
                    label=feature.replace("_", " ").title(),
                    value=f"PSI: {psi:.4f}",
                    delta=status_label,
                    delta_color="off"
                )

    st.divider()

    # Distribution comparison charts
    st.markdown("**Distribution Comparison — Baseline vs Incoming Orders**")

    for feature, col_name in features_to_monitor.items():
        if feature not in psi_results:
            continue

        st.markdown(f"**{feature.replace('_', ' ').title()}**")

        baseline = BASELINE_DISTRIBUTIONS[feature]
        current = psi_results[feature]["current"]

        # Align keys across both distributions
        all_keys = sorted(set(list(baseline.keys()) + list(current.keys())), key=str)

        baseline_vals = [baseline.get(k, 0) for k in all_keys]
        current_vals = [current.get(k, 0) for k in all_keys]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Training Baseline",
            x=[str(k) for k in all_keys],
            y=baseline_vals,
            marker_color="#4a90d9",
            opacity=0.8
        ))

        fig.add_trace(go.Bar(
            name="Incoming Orders",
            x=[str(k) for k in all_keys],
            y=current_vals,
            marker_color="#e8513a",
            opacity=0.8
        ))

        fig.update_layout(
            barmode="group",
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_tickformat=".0%"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("**What to do if drift is detected**")
    st.markdown("""
    - **Moderate drift** — monitor closely, no immediate action required
    - **Significant drift** — investigate whether order patterns have genuinely changed
    - If drift persists, retrain the model on more recent order data and redeploy
    """)