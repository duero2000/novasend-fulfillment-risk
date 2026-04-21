# model_performance.py
# Model Performance tab — displays key metrics, confusion matrix, and feature importance
# All values are hardcoded from the final evaluation run since Databricks Community Edition
# does not expose MLflow metrics via public API without authentication

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# Final evaluation metrics from the tuned XGBoost model
MODEL_METRICS = {
    "AUC": 0.7841,
    "F1 Score": 0.6920,
    "Precision": 0.85,
    "Recall": 0.58
}

# Confusion matrix values from the test set evaluation
CONFUSION_MATRIX = {
    "TP": 11730,
    "FP": 2070,
    "FN": 8270,
    "TN": 14080
}

# SHAP-based feature importance from the evaluation notebook
FEATURE_IMPORTANCE = {
    "Days_for_shipment_scheduled": 1.67,
    "Type_TRANSFER": 0.85,
    "Order_Status_PENDING": 0.43,
    "Order_Status_PROCESSING": 0.38,
    "Shipping_Mode_Same_Day": 0.34,
    "discount_tier_aggressive": 0.28,
    "region_late_rate": 0.21,
    "Market_LATAM": 0.18,
    "order_month": 0.15,
    "order_quarter": 0.13,
    "Order_Status_ON_HOLD": 0.11,
    "Customer_Segment_Corporate": 0.09,
    "Market_USCA": 0.08,
    "Order_Status_CLOSED": 0.06,
    "Order_Status_SUSPECTED_FRAUD": 0.04
}


def render_model_performance():
    st.subheader("📊 Model Performance")
    st.write("Evaluation results for the tuned XGBoost model on the held-out test set.")

    # Top level metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC", f"{MODEL_METRICS['AUC']:.4f}", help="Area under the ROC curve")
    col2.metric("F1 Score", f"{MODEL_METRICS['F1 Score']:.4f}", help="Harmonic mean of precision and recall")
    col3.metric("Precision", f"{MODEL_METRICS['Precision']:.2f}", help="Of predicted late orders, how many were actually late")
    col4.metric("Recall", f"{MODEL_METRICS['Recall']:.2f}", help="Of actual late orders, how many did the model catch")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Confusion Matrix**")

        # Build confusion matrix as a heatmap
        cm_data = [
            [CONFUSION_MATRIX["TN"], CONFUSION_MATRIX["FP"]],
            [CONFUSION_MATRIX["FN"], CONFUSION_MATRIX["TP"]]
        ]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted On Time", "Predicted Late"],
            y=["Actually On Time", "Actually Late"],
            colorscale="Blues",
            text=[
                [f"TN: {CONFUSION_MATRIX['TN']:,}", f"FP: {CONFUSION_MATRIX['FP']:,}"],
                [f"FN: {CONFUSION_MATRIX['FN']:,}", f"TP: {CONFUSION_MATRIX['TP']:,}"]
            ],
            texttemplate="%{text}",
            showscale=False
        ))

        fig_cm.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=20, r=20)
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.markdown("**SHAP Feature Importance**")

        # Sort features by importance descending
        importance_df = pd.DataFrame(
            list(FEATURE_IMPORTANCE.items()),
            columns=["Feature", "Mean SHAP Value"]
        ).sort_values("Mean SHAP Value", ascending=True)

        fig_shap = px.bar(
            importance_df,
            x="Mean SHAP Value",
            y="Feature",
            orientation="h",
            color="Mean SHAP Value",
            color_continuous_scale="Blues"
        )

        fig_shap.update_layout(
            height=450,
            margin=dict(t=20, b=20, l=20, r=20),
            coloraxis_showscale=False
        )

        st.plotly_chart(fig_shap, use_container_width=True)

    st.divider()

    st.markdown("**Model Details**")
    details = {
        "Model": "XGBoost",
        "Features": "15 (stepwise selected)",
        "Training Rows": "144,415",
        "Test Rows": "36,104",
        "n_estimators": 700,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8
    }

    details_df = pd.DataFrame(list(details.items()), columns=["Parameter", "Value"])
    st.dataframe(details_df, use_container_width=True, hide_index=True)