# main.py
# FastAPI prediction API for NovaSend fulfillment risk scoring
# Loads the tuned XGBoost model from Databricks MLflow at startup
from dotenv import load_dotenv
load_dotenv()  # Loads .env into environment before MLflow reads credentials

import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from api.schema import OrderInput

# Initialize the FastAPI app
app = FastAPI(
    title="NovaSend Fulfillment Risk API",
    description="Scores active orders by late delivery risk probability",
    version="1.0.0"
)

# MLflow tracking URI points to Databricks
mlflow.set_tracking_uri("databricks")

# Model is loaded once at module load time to avoid cold-start latency on each request

# Load directly from run artifact path — bypasses Unity Catalog permission layer
MODEL_URI = os.getenv(
    "MODEL_URI",
    "runs:/6962f2573f114a089faccb7c5829a95c/xgboost_tuned_model"
)

model = mlflow.sklearn.load_model(MODEL_URI)

def encode_order(order: OrderInput) -> pd.DataFrame:
    # Build the 15 features the model expects from human-readable API inputs
    return pd.DataFrame([{
        # Pass-through numerics — no transformation needed
        "Days_for_shipment_scheduled": order.days_for_shipment_scheduled,
        "order_month":                 order.order_month,
        "order_quarter":               order.order_quarter,
        "region_late_rate":            order.region_late_rate,

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

@app.post("/predict")
def predict(order: OrderInput):
    try:
        # Encode incoming order into the 15-feature DataFrame the model expects
        features = encode_order(order)

        # Returns probability for both classes — index 1 is late delivery probability
        prob = model.predict_proba(features)[0][1]

        # Bucket raw probability into an operational risk tier
        if prob >= 0.75:
            risk_tier = "HIGH"
        elif prob >= 0.50:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        return {
            "late_delivery_probability": round(float(prob), 4),
            "risk_tier": risk_tier
        }

    except Exception as e:
        # Surface the error to the caller rather than returning a silent 500
        raise HTTPException(status_code=500, detail=str(e))
