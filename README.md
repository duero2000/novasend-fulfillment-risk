# NovaSend Fulfillment Risk — ML Pipeline

### Predicting late delivery risk per order at intake across 180K supply chain records

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://novasend-fulfillment-risk.streamlit.app/)

---

## Responsible AI Use

This project was developed with assistance from Claude, an AI assistant by Anthropic, used as a coding partner throughout the build. All modeling decisions, feature engineering logic, analytical judgments, and business framing were made by me and grounded in my own understanding of supply chain operations and fulfillment risk. AI assistance was used solely to accelerate development and validate thinking on occasion, not to replace it. The DataCo Supply Chain dataset is a publicly available research dataset and contains no real customer data.

---

## Live Links

| Resource | URL |
|---|---|
| Operations Dashboard | https://novasend-fulfillment-risk.streamlit.app/ |

---

## Business Problem

NovaSend Logistics is a fictional mid-sized U.S. e-commerce fulfillment company handling ~40,000 orders per day across six regional fulfillment centers. Over the past 12 months, NovaSend experienced a 17% late delivery rate, up from 11% two years prior.

Late deliveries were identified reactively, after a customer complaint or missed SLA, leaving no room for intervention. At $18 per SLA penalty, the estimated annual impact was $4.2M.

This pipeline predicts late delivery probability per order at intake, ranks orders by risk tier, and surfaces a prioritized intervention queue for operations supervisors before shipments leave the warehouse. Intervening on 30% of predicted high-risk orders projects to recover approximately $1.1M annually in avoided SLA penalties alone.

---

## Architecture

```
DataCo Dataset (180K rows)
        │
        ▼
Databricks Community Edition
├── Raw Delta table (novasend_fulfillment.raw)
├── EDA notebook (01_eda.ipynb)
├── Feature engineering notebook (02_feature_engineering.ipynb)
│   └── Processed Delta table (novasend_fulfillment.processed.features_engineered)
├── Modeling notebook (03_modeling.ipynb)
│   └── MLflow experiment tracking (/novasend-fulfillment-risk)
└── Model artifact (models/xgboost_tuned_model.pkl)
        │
        ▼
FastAPI Prediction API (Render)
├── GET  /health
└── POST /predict → risk_score + risk_tier (LOW / MEDIUM / HIGH / CRITICAL)
        │
        ▼
Streamlit Operations Dashboard (Streamlit Community Cloud)
├── Risk Queue — live scored orders ranked by tier
├── Model Performance — metrics, confusion matrix, SHAP importance
├── Experiment Comparison — all 6 MLflow runs visualized
└── Drift Monitoring — PSI across 4 monitored features
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data platform | Databricks Community Edition, Delta Lake |
| Modeling | Python, Pandas, Scikit-learn, XGBoost, LightGBM |
| Experiment tracking | Databricks MLflow |
| Prediction API | FastAPI, Uvicorn, joblib |
| Deployment | Render (GitHub-native, no Docker) |
| Dashboard | Streamlit Community Cloud |
| Explainability | SHAP |
| Version control | GitHub |

---

## Model Results

| Model | AUC | F1 |
|---|---|---|
| XGBoost baseline | 0.7802 | 0.6835 |
| LightGBM baseline | 0.7792 | 0.6824 |
| XGBoost tuned | **0.7841** | **0.6920** |

The final model uses 15 features selected from an original set of 35 via Sequential Feature Selector. The leaner feature set performed comparably to the full set while being faster to serve and easier to explain.

### Selected Features (15)

`Days_for_shipment_scheduled` · `order_month` · `order_quarter` · `region_late_rate` · `Type_TRANSFER` · `Customer_Segment_Corporate` · `Market_LATAM` · `Market_USCA` · `Order_Status_CLOSED` · `Order_Status_ON_HOLD` · `Order_Status_PENDING` · `Order_Status_PROCESSING` · `Order_Status_SUSPECTED_FRAUD` · `Shipping_Mode_Same_Day` · `discount_tier_aggressive`

### Best XGBoost Parameters

| Parameter | Value |
|---|---|
| `n_estimators` | 700 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.8 |

---

## Repo Structure

```
novasend-fulfillment-risk/
├── data/
│   ├── raw/                        # Excluded from GitHub — Delta tables only
│   └── processed/                  # Excluded from GitHub — Delta tables only
├── models/
│   └── xgboost_tuned_model.pkl     # Committed model artifact
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_drift_monitoring.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── api/
│   ├── main.py
│   ├── schema.py
│   └── requirements.txt
├── dashboard/
│   ├── app.py
│   ├── tabs/
│   │   ├── risk_queue.py
│   │   ├── model_performance.py
│   │   ├── experiment_design.py
│   │   └── drift_monitoring.py
│   └── requirements.txt
├── reports/
│   └── case_study.md
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### Running the API locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Send a POST request to `/predict` with a JSON body matching the schema in `api/schema.py`.

### Running the dashboard locally

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard expects the API to be running. By default it points to the live Render deployment. To point it at a local API instance, update the `API_URL` variable in `app.py`.

### Databricks notebooks

The notebooks in `notebooks/` are designed to run in Databricks Community Edition. Raw data is not included in this repo. To reproduce the pipeline:

1. Download the DataCo Supply Chain dataset from Kaggle
2. Upload the CSV to Databricks DBFS
3. Run `01_eda.ipynb` to produce the cleaned Delta table
4. Run `02_feature_engineering.ipynb` to produce the engineered features Delta table
5. Run `03_modeling.ipynb` to train, tune, and export the model

---

## Key Design Decisions

**Class distribution.** The dataset provided a cleaner starting point than the business narrative suggested. With a 55% late delivery rate observed in the data, this was never a rare event problem. That distribution gave the model a strong, balanced representation to learn from without any intervention needed.

**Joblib over MLflow registry at runtime.** Databricks Community Edition PAT token scopes blocked Unity Catalog model registry promotion. Loading from the committed `.pkl` file keeps the API self-contained and eliminates a live Databricks dependency in production.

**Sequential feature selection over full feature set.** Dropping from 35 to 15 features via SFS produced a model that performed comparably to the full set with better interpretability and lower serving latency.

**PSI approach for binary features.** Histogram-based PSI breaks down at low prevalence values. Binary drift features use proportion comparison instead, which is more stable and more honest at small sample sizes.

**API wake-up gate.** Render's free tier introduces cold start latency of 30 to 60 seconds after idle periods. The dashboard pings the health endpoint on load and holds all tabs until the service confirms it is awake.

**No Docker.** Render deploys natively from GitHub. Docker was not needed and was deliberately excluded to keep the deployment simple and portable.

---

## Leakage Controls

The following columns were excluded from all feature sets:

| Column | Reason |
|---|---|
| `Days for shipping (real)` | Actual delivery duration — recorded post-delivery |
| `Delivery Status` | Verbose restatement of the target label |
| `Benefit per order` | Recorded post-fulfillment |
| `Order Profit Per Order` | Recorded post-fulfillment |
| `Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname` | PII — no predictive value |

---

## Dataset

**Source**: DataCo Supply Chain Dataset — Kaggle  
**Size**: 180,519 order-level records  
**Target variable**: `Late_delivery_risk` (binary: 1 = late, 0 = on time)  
**Class distribution**: ~55% late, ~45% on time  
**Raw data**: Not committed to GitHub — loaded into Databricks as a Delta table
