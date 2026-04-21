# experiment_design.py
# Experiment Comparison tab — displays all model runs from the MLflow experiment
# Values are hardcoded from Databricks MLflow since Community Edition does not
# expose a public tracking API without authentication

import streamlit as st
import pandas as pd
import plotly.express as px


# All tracked runs from the /novasend-fulfillment-risk MLflow experiment
EXPERIMENT_RUNS = [
    {
        "Run Name": "logistic_regression_baseline",
        "Model": "Logistic Regression",
        "Features": 15,
        "AUC": 0.7695,
        "F1": 0.6724,
        "Notes": "Baseline — scaled features, no regularization"
    },
    {
        "Run Name": "xgboost_baseline",
        "Model": "XGBoost",
        "Features": 34,
        "AUC": 0.7802,
        "F1": 0.6835,
        "Notes": "Full feature set, default params"
    },
    {
        "Run Name": "lightgbm_baseline",
        "Model": "LightGBM",
        "Features": 34,
        "AUC": 0.7792,
        "F1": 0.6824,
        "Notes": "Full feature set, default params"
    },
    {
        "Run Name": "xgboost_reduced",
        "Model": "XGBoost",
        "Features": 15,
        "AUC": 0.7813,
        "F1": 0.6846,
        "Notes": "Stepwise selected features, default params"
    },
    {
        "Run Name": "lightgbm_reduced",
        "Model": "LightGBM",
        "Features": 15,
        "AUC": 0.7795,
        "F1": 0.6837,
        "Notes": "Stepwise selected features, default params"
    },
    {
        "Run Name": "xgboost_tuned",
        "Model": "XGBoost",
        "Features": 15,
        "AUC": 0.7841,
        "F1": 0.6920,
        "Notes": "Tuned — n_estimators=700, max_depth=6, lr=0.05"
    }
]



def render_experiment_design():
    st.subheader("🧪 Experiment Comparison")
    st.write("All model runs tracked in the Databricks MLflow experiment `/novasend-fulfillment-risk`.")

    df = pd.DataFrame(EXPERIMENT_RUNS)
    
    # Sort by AUC descending so the winning model appears at the top
    df = df.sort_values("AUC", ascending=False).reset_index(drop=True)

    # Highlight the winning model
    st.markdown("**All Runs**")

    def highlight_winner(row):
        # Gold background for the tuned XGBoost which is the selected model
        if row["Run Name"] == "xgboost_tuned":
            return ["background-color: #fff9c4"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(highlight_winner, axis=1).format({
        "AUC": "{:.4f}",
        "F1": "{:.4f}"
    })

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**AUC by Run**")

        fig_auc = px.bar(
            df,
            x="Run Name",
            y="AUC",
            color="Model",
            text="AUC",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig_auc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_auc.update_layout(
            height=400,
            xaxis_tickangle=45,
            margin=dict(t=20, b=80, l=20, r=20),
            yaxis=dict(range=[0.75, 0.80]),
            showlegend=True
        )

        st.plotly_chart(fig_auc, use_container_width=True)

    with col_right:
        st.markdown("**F1 by Run**")

        fig_f1 = px.bar(
            df,
            x="Run Name",
            y="F1",
            color="Model",
            text="F1",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig_f1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_f1.update_layout(
            height=400,
            xaxis_tickangle=45,
            margin=dict(t=20, b=80, l=20, r=20),
            yaxis=dict(range=[0.66, 0.70]),
            showlegend=True
        )

        st.plotly_chart(fig_f1, use_container_width=True)

    st.divider()

    st.markdown("**Key Takeaways**")
    st.markdown("""
    - Stepwise feature selection improved XGBoost AUC from **0.7802 → 0.7813** while reducing features from 34 to 15
    - Hyperparameter tuning pushed XGBoost to **0.7841 AUC** — the best result across all runs
    - LightGBM and XGBoost performed nearly identically at baseline — XGBoost was selected for tuning
    - Logistic Regression at **0.7695 AUC** confirms the problem has meaningful signal beyond a linear boundary
    """)