# NovaSend Fulfillment Risk — Case Study

## Executive Summary

I built an end-to-end machine learning pipeline to predict late delivery risk per order at intake for NovaSend Logistics, a fictional mid-sized U.S. e-commerce fulfillment company. The business problem was a 17% late delivery rate generating an estimated $4.2M in annual losses through SLA penalties, elevated customer service volume, and retailer churn. The final pipeline includes a tuned XGBoost classifier (AUC 0.7841), a live FastAPI prediction endpoint deployed on Render, and a Streamlit operations dashboard with real-time synthetic order scoring and drift monitoring.

---

## Business Problem

NovaSend had no systematic way to identify high-risk orders before they left the warehouse. Dispatchers relied on intuition, a process that does not scale and misses the majority of at-risk shipments before intervention is still possible.

The cost structure made the problem concrete:

- $18 SLA penalty per late order
- Elevated customer service volume and return rates downstream
- Retailer churn from repeated delivery failures
- Total estimated annual impact: $4.2M

The proposed solution was a model that scores every incoming order by late delivery probability at intake, ranks orders by risk tier, and surfaces a prioritized intervention queue for operations supervisors before shipments leave the facility. If the operations team intervenes on 30% of predicted high-risk orders and converts them to on-time deliveries, the projected annual recovery is $1.1M in avoided SLA penalties alone.

---

## Dataset and Data Pipeline

The dataset is the DataCo Supply Chain dataset from Kaggle, with 180,519 order records spanning markets in North America, Latin America, Europe, Africa, and Pacific Asia. The target variable is `Late_delivery_risk`, a simple binary label: 1 means the order arrived late, 0 means it did not. About 55% of orders were late and 45% were on time.

That class split was close enough to even that oversampling was not needed. Some projects default to techniques like SMOTE whenever the classes are not perfectly balanced, but a 55/45 split is not a real imbalance problem. Adding oversampling would have introduced unnecessary complexity without solving anything.

Before any modeling, I identified and removed columns that would have caused data leakage. Leakage happens when a feature contains information that would not be available at the time of prediction in the real world. Three categories of columns were excluded:

- Post-delivery measurements like actual shipping duration and delivery status
- Post-fulfillment financials like profit per order
- Customer PII like name, email, and password, which have no predictive value and no place in a model

Raw data lives in Databricks as a Delta table and is not committed to GitHub.

---

## Feature Engineering

I started with 23 cleaned columns and built out a final set of 35 features before moving to model selection. Here is what was added and why.

**Temporal features.** The order date was broken into day of week, month, quarter, and a weekend flag. Fulfillment operations run differently on different days and at different points in the year, so these capture that rhythm.

**Log transformations.** Price and discount columns were right-skewed, meaning a small number of very large values were pulling the distribution. Log transforming them compresses that tail and gives the model a cleaner signal.

**Discount tiers.** Instead of feeding the raw discount rate into the model, I bucketed it into four categories: none, low, medium, and aggressive. This made the feature easier to interpret and preserved the signal without overfitting to exact decimal values.

**Region encoding.** Order region had over 20 unique values. Rather than one-hot encoding all of them, I replaced the column with each region's historical late delivery rate, calculated using 5-fold cross-validation. The cross-validation step is important because it prevents the target variable from leaking into the feature during training.

One feature I decided not to build was a shipping mode and region interaction. EDA showed that late delivery rates were nearly identical across all regions, with almost no variance. Combining a strong predictor like shipping mode with a near-flat variable like region would have added noise rather than signal, so I left it out.

---

## Modeling

With 35 engineered features in hand, my first instinct was not to throw all of them at a model. More features is not always better. Irrelevant features add noise, slow down training, and make models harder to explain to a business audience. I wanted to find the smallest set of features that still delivered strong predictive performance.

**Feature Selection**

I used Sequential Feature Selector from scikit-learn to systematically evaluate which features were actually pulling weight. The process works by starting with no features and adding them one at a time, keeping only the ones that improve cross-validated AUC at each step. After running selection across both XGBoost and LightGBM, 15 features were retained from the original 35.

The final 15 features were:

- `Days_for_shipment_scheduled`
- `order_month`, `order_quarter`
- `region_late_rate`
- `Type_TRANSFER`
- `Customer_Segment_Corporate`
- `Market_LATAM`, `Market_USCA`
- `Order_Status_CLOSED`, `Order_Status_ON_HOLD`, `Order_Status_PENDING`, `Order_Status_PROCESSING`, `Order_Status_SUSPECTED_FRAUD`
- `Shipping_Mode_Same_Day`
- `discount_tier_aggressive`

The strongest single predictor was `Days_for_shipment_scheduled`, which had a correlation of 0.37 with the target. `Shipping_Mode_Same_Day` was the dominant categorical signal, with First Class sitting at roughly a 95% late rate in EDA.

**Baseline Models**

I ran baseline XGBoost and LightGBM models before any tuning to establish a performance floor.

| Model | AUC | F1 |
|---|---|---|
| XGBoost baseline | 0.7802 | 0.6835 |
| LightGBM baseline | 0.7792 | 0.6824 |

Both models were close to the 0.80 AUC target but not there yet.

**Hyperparameter Tuning**

I used RandomizedSearchCV to tune XGBoost across key parameters. The winning configuration was:

| Parameter | Value |
|---|---|
| `n_estimators` | 700 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.8 |

Tuned AUC landed at 0.7841 with an F1 of 0.6920. The model did not clear 0.80, but it closed the gap meaningfully. More importantly, the 15-feature tuned model performed comparably to running all 35 features through the same algorithm. That outcome validated the variable selection step: a simpler, more interpretable model with no meaningful loss in predictive power.

All six experiment runs were tracked in Databricks MLflow under the experiment `/novasend-fulfillment-risk`. The final model was exported as `models/xgboost_tuned_model.pkl` and committed to the GitHub repository.

---

## Production Infrastructure

Building a model that lives in a notebook is not enough for a portfolio that targets production-oriented roles. I wanted the pipeline to have real infrastructure: a live API that scores orders on demand and a dashboard that an operations team could actually use.

**Prediction API**

The prediction API is a FastAPI application deployed on Render. It exposes two endpoints:

- `GET /health` — confirms the service is alive
- `POST /predict` — accepts order features and returns a risk score and risk tier

The risk tier buckets the raw probability output into four actionable categories: LOW, MEDIUM, HIGH, and CRITICAL. This translation matters because a number like 0.73 is not immediately useful to an operations supervisor, but CRITICAL is.

The model loads from the committed `xgboost_tuned_model.pkl` file using joblib at API startup. I made a deliberate decision not to load the model from the Databricks MLflow registry at runtime. Databricks Community Edition PAT token scopes blocked Unity Catalog model registry promotion, and introducing a live Databricks dependency into the API would have made the deployment fragile. Loading from the repo directly keeps the API self-contained and reliable.

Render deploys natively from GitHub with no Docker required. The root directory is set to `api` and the start command is `uvicorn main:app --host 0.0.0.0 --port $PORT`.

**Streamlit Dashboard**

The operations dashboard is deployed on Streamlit Community Cloud and organized into four pages:

1. Risk Queue — displays scored orders ranked by risk tier
2. Model Performance — hardcoded metrics, confusion matrix heatmap, and SHAP feature importance
3. Experiment Comparison — all six MLflow runs visualized with AUC and F1 bar charts
4. Drift Monitoring — PSI-based feature drift detection across four monitored features

One practical challenge with Render's free tier is cold starts. When the service has been idle, the first request can take 30 to 60 seconds to respond. I handled this with an API wake-up gate in `app.py` that pings the health endpoint on dashboard load and blocks all tabs until the service confirms it is awake. This prevents the dashboard from showing errors or stale data during a cold start.

The dashboard also includes a synthetic order generator. A single button creates a realistic randomized order, scores it through the live API, and populates the risk queue in real time. This was important for the portfolio because it lets anyone interact with the full pipeline without needing access to real order data.

---

## Drift Monitoring

A model that performs well at training time can degrade silently in production. Order patterns shift, customer behavior changes, and the relationships the model learned can become stale. Drift monitoring is how you catch that before it becomes a business problem.

**What Is Drift**

Feature drift happens when the distribution of incoming data starts to look different from the data the model was trained on. The model was never told about this new distribution, so its predictions become less reliable without any visible error or warning.

**What I Monitored**

I selected four features to monitor based on their importance to the model and their likelihood of shifting in a real operational environment:

- `Days_for_shipment_scheduled` — the strongest predictor in the model
- `Order_Status_SUSPECTED_FRAUD` — a low-prevalence binary flag that could spike during fraud events
- `Shipping_Mode_Same_Day` — a high-signal categorical feature tied to fulfillment capacity
- `discount_tier_aggressive` — a business-controlled lever that could change with promotions

**How Drift Is Measured**

I used Population Stability Index (PSI) to quantify drift. PSI compares the distribution of a feature in production against the training baseline and returns a score that falls into one of three ranges:

| PSI Score | Interpretation |
|---|---|
| Below 0.10 | No meaningful drift |
| 0.10 to 0.25 | Moderate drift, worth monitoring |
| Above 0.25 | Significant drift, investigate |

For continuous features like `Days_for_shipment_scheduled`, PSI uses histogram binning to compare distributions. For binary features, I used a simpler proportion comparison rather than histogram binning. Histogram-based PSI breaks down at low prevalence values, which is exactly the situation with a rare flag like `Order_Status_SUSPECTED_FRAUD`. Proportion comparison is more stable and more honest at small sample sizes.

**Findings**

In the simulated production sample, `Days_for_shipment_scheduled` and `Order_Status_SUSPECTED_FRAUD` showed meaningful drift, while `Shipping_Mode_Same_Day` and `discount_tier_aggressive` remained stable. This outcome reflects a realistic scenario where fulfillment scheduling and fraud patterns shift over time while shipping mode mix and discount strategy stay relatively consistent.

The drift monitoring tab in the Streamlit dashboard runs the same PSI calculations live against the current risk queue, giving operations teams a real-time signal when incoming orders start looking different from what the model was trained on.

---

## Results and Business Impact

The tuned XGBoost model achieved an AUC of 0.7841 and an F1 of 0.6920 on held-out test data. It did not clear the original 0.80 AUC target, but it came close and did so with a leaner, more interpretable feature set than the baseline.

The more important result is what the model enables operationally. Before this pipeline, NovaSend had no way to identify high-risk orders before they left the warehouse. Dispatchers were working reactively, catching problems only after a customer complaint or missed SLA. This pipeline changes that dynamic by scoring every order at intake and surfacing the highest-risk shipments before intervention is no longer possible.

The business impact projection is based on a conservative assumption: if the operations team intervenes on 30% of predicted high-risk orders and converts them to on-time deliveries, the avoided SLA penalties alone recover approximately $1.1M annually against a $4.2M problem. That number does not include downstream savings from reduced customer service volume, lower return rates, or improved retailer retention.

The full pipeline is live. Orders can be scored through the API in real time, the dashboard surfaces a prioritized risk queue without manual intervention, and drift monitoring provides an early warning system for model degradation.

---

## Lessons Learned

**Simpler models earn their place.** Going from 35 features to 15 through sequential feature selection produced a model that performed comparably to the full feature set. The leaner model is faster, easier to explain, and easier to maintain. The lesson is that variable selection is not just a performance optimization, it is a discipline.

**Infrastructure decisions are modeling decisions.** Choosing to load the model from joblib instead of the MLflow registry was not a workaround, it was the right call given the constraints of Databricks Community Edition. Every production environment has constraints, and working within them cleanly is a skill.

**Leakage prevention requires deliberate upfront thinking.** Several columns in the raw dataset looked like useful features until you asked whether they would actually be available at the time of prediction. Building that habit early saved the model from being built on a false foundation.

**PSI is not one size fits all.** Applying histogram-based PSI to a binary feature with low prevalence produces unstable and misleading results. Matching the measurement approach to the feature type is as important as the measurement itself.

**A live demo changes the conversation.** The synthetic order generator in the dashboard lets anyone interact with the full pipeline without needing real data access. In a portfolio context, showing a working system is more compelling than describing one.
