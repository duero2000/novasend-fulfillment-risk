# main.py
# FastAPI prediction API for NovaSend fulfillment risk scoring
# Loads the tuned XGBoost model from a local .pkl file committed to the repo

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from schema import OrderInput

# Initialize the FastAPI app
app = FastAPI(
    title="NovaSend Fulfillment Risk API",
    description="Scores active orders by late delivery risk probability",
    version="1.0.0"
)

# Load model once at startup from the models/ folder in the repo
# Path is relative to the repo root so it works on any machine or server
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_tuned_model.pkl")
model = joblib.load(MODEL_PATH)


def encode_order(order: OrderInput) -> pd.DataFrame:
    # Build the 15 features the model expects from human-readable API inputs
    return pd.DataFrame([{
        # Pass-through numerics — no transformation needed
        "Days_for_shipment_scheduled":  order.days_for_shipment_scheduled,
        "order_month":                  order.order_month,
        "order_quarter":                order.order_quarter,
        "region_late_rate":             order.region_late_rate,

        # Payment type OHE
        "Type_TRANSFER": int(order.payment_type == "TRANSFER"),

        # Customer segment OHE
        "Customer_Segment_Corporate": int(order.customer_segment == "Corporate"),

        # Market OHE
        "Market_LATAM": int(order.market == "LATAM"),
        "Market_USCA":  int(order.market == "USCA"),

        # Order status OHE
        "Order_Status_CLOSED":          int(order.order_status == "CLOSED"),
        "Order_Status_ON_HOLD":         int(order.order_status == "ON_HOLD"),
        "Order_Status_PENDING":         int(order.order_status == "PENDING"),
        "Order_Status_PROCESSING":      int(order.order_status == "PROCESSING"),
        "Order_Status_SUSPECTED_FRAUD": int(order.order_status == "SUSPECTED_FRAUD"),

        # Shipping mode OHE
        "Shipping_Mode_Same_Day": int(order.shipping_mode == "Same Day"),

        # Discount tier — aggressive if rate exceeds 0.2
        "discount_tier_aggressive": int(order.order_item_discount_rate > 0.2),
    }])


@app.get("/health")
def health():
    # Confirms the API is running and the model loaded successfully
    return {"status": "ok", "model": "xgboost_tuned", "version": "1.0.0"}


@app.post("/predict")
def predict(order: OrderInput):
    try:
        # Encode the incoming order into the 15-feature DataFrame the model expects
        X = encode_order(order)

        # Get the late delivery probability from the model
        risk_score = float(model.predict_proba(X)[0][1])

        # Assign a risk tier based on the probability score
        if risk_score >= 0.75:
            risk_tier = "CRITICAL"
        elif risk_score >= 0.55:
            risk_tier = "HIGH"
        elif risk_score >= 0.35:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        return {
            "risk_score": round(risk_score, 4),
            "risk_tier": risk_tier
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))