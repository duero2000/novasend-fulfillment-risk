# app.py
# Main entry point for the NovaSend Fulfillment Risk Streamlit dashboard

import streamlit as st
import requests

from tabs.risk_queue import render_risk_queue
from tabs.model_performance import render_model_performance
from tabs.experiment_design import render_experiment_design
from tabs.drift_monitoring import render_drift_monitoring

st.set_page_config(
    page_title="NovaSend Fulfillment Risk Dashboard",
    page_icon="📦",
    layout="wide"
)


def wake_api() -> bool:
    try:
        response = requests.get(
            "https://novasend-fulfillment-risk.onrender.com/health",
            timeout=120
        )
        return response.status_code == 200
    except Exception:
        return False


if "api_ready" not in st.session_state:
    st.session_state.api_ready = False

if not st.session_state.api_ready:
    st.title("NovaSend Fulfillment Risk Dashboard")
    st.warning("Connecting to the prediction API. This may take up to 90 seconds on first load.")
    progress = st.progress(0, text="Waking up the API server...")

    for i in range(1, 4):
        progress.progress(i * 25, text=f"Attempting to reach API... (attempt {i} of 3)")
        if wake_api():
            st.session_state.api_ready = True
            progress.progress(100, text="API is ready.")
            break

    if not st.session_state.api_ready:
        st.error(
            "The prediction API could not be reached after 3 attempts. "
            "Please refresh the page to try again."
        )
        st.stop()
    else:
        st.rerun()

# Sidebar navigation
st.sidebar.title("NovaSend")
st.sidebar.caption("Fulfillment Risk Dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Risk Queue", "Model Performance", "Experiment Comparison", "Drift Monitoring"]
)

st.sidebar.divider()
st.sidebar.caption("Prediction API")
st.sidebar.success("Connected")

# Render the selected page
if page == "Risk Queue":
    render_risk_queue()
elif page == "Model Performance":
    render_model_performance()
elif page == "Experiment Comparison":
    render_experiment_design()
elif page == "Drift Monitoring":
    render_drift_monitoring()