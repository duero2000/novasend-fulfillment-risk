# risk_queue.py
# Risk Queue tab — generates and scores synthetic orders, surfaces only at-risk orders
# for operations supervisor review. Order detail opens in a st.dialog popup.

import streamlit as st
import requests
import random
import pandas as pd
from datetime import datetime

API_URL = "https://novasend-fulfillment-risk.onrender.com/predict"

# ── Reference data pools ──────────────────────────────────────────────────────

SHIPPING_MODES     = ["Standard Class", "First Class", "Second Class", "Same Day"]
CUSTOMER_SEGMENTS  = ["Consumer", "Corporate", "Home Office"]
PAYMENT_TYPES      = ["DEBIT", "TRANSFER", "CASH", "PAYMENT"]
ORDER_STATUSES     = ["PENDING", "PROCESSING", "CLOSED", "ON_HOLD", "SUSPECTED_FRAUD", "COMPLETE"]

STATUS_LABELS = {
    "PENDING":          "Pending",
    "PROCESSING":       "Processing",
    "CLOSED":           "Closed",
    "ON_HOLD":          "On Hold",
    "SUSPECTED_FRAUD":  "Suspected Fraud",
    "COMPLETE":         "Complete",
}

# Market -> list of (region, city, country) tuples
# Keeps geography internally consistent so a LATAM order never shows an Asian city
MARKET_GEO = {
    "USCA": [
        ("Western US",  "Los Angeles",  "United States"),
        ("Western US",  "San Jose",     "United States"),
        ("Eastern US",  "Miami",        "United States"),
        ("Eastern US",  "Tonawanda",    "United States"),
        ("Southern US", "Houston",      "United States"),
        ("Eastern US",  "Chicago",      "United States"),
        ("Canada",      "Toronto",      "Canada"),
        ("Canada",      "Vancouver",    "Canada"),
    ],
    "LATAM": [
        ("Caribbean",       "San Juan",     "Puerto Rico"),
        ("South America",   "Bogota",       "Colombia"),
        ("Central America", "Mexico City",  "Mexico"),
        ("South America",   "Buenos Aires", "Argentina"),
        ("South America",   "Lima",         "Peru"),
        ("Central America", "Tegucigalpa",  "Honduras"),
        ("South America",   "Santiago",     "Chile"),
        ("Caribbean",       "Havana",       "Cuba"),
    ],
    "Europe": [
        ("Western Europe",  "Paris",      "France"),
        ("Western Europe",  "Berlin",     "Germany"),
        ("Southern Europe", "Madrid",     "Spain"),
        ("Southern Europe", "Rome",       "Italy"),
        ("Western Europe",  "Amsterdam",  "Netherlands"),
        ("Eastern Europe",  "Warsaw",     "Poland"),
        ("Northern Europe", "Stockholm",  "Sweden"),
        ("Southern Europe", "Lisbon",     "Portugal"),
    ],
    "Africa": [
        ("West Africa",     "Lagos",        "Nigeria"),
        ("East Africa",     "Nairobi",      "Kenya"),
        ("North Africa",    "Cairo",        "Egypt"),
        ("Southern Africa", "Johannesburg", "South Africa"),
        ("West Africa",     "Accra",        "Ghana"),
        ("Central Africa",  "Kinshasa",     "DR Congo"),
        ("North Africa",    "Casablanca",   "Morocco"),
        ("East Africa",     "Addis Ababa",  "Ethiopia"),
    ],
    "Pacific Asia": [
        ("Southeast Asia", "Bekasi",    "Indonesia"),
        ("Eastern Asia",   "Guangzhou", "China"),
        ("South Asia",     "Mumbai",    "India"),
        ("Oceania",        "Sydney",    "Australia"),
        ("Southeast Asia", "Bangkok",   "Thailand"),
        ("Eastern Asia",   "Tokyo",     "Japan"),
        ("South Asia",     "Karachi",   "Pakistan"),
        ("Oceania",        "Melbourne", "Australia"),
    ],
}

# Department -> list of (product_name, price_min, price_max)
DEPARTMENT_PRODUCTS = {
    "Fitness":     [("Smart Watch", 280, 350), ("Yoga Mat", 25, 60), ("Resistance Bands", 15, 40)],
    "Apparel":     [("Running Shoes", 80, 160), ("Winter Jacket", 90, 200), ("Sports Jersey", 40, 90)],
    "Electronics": [("Bluetooth Speaker", 50, 120), ("Laptop Stand", 30, 80), ("USB Hub", 20, 60)],
    "Garden":      [("Garden Hose", 25, 70), ("Patio Chair", 80, 180), ("Outdoor Grill", 150, 400)],
    "Fan Shop":    [("Team Cap", 20, 45), ("Sports Mug", 12, 30), ("Wall Flag", 15, 40)],
}

# Risk tiers that require supervisor attention
AT_RISK_TIERS = {"CRITICAL", "HIGH", "MEDIUM"}

# Tier badge colors
TIER_COLORS = {
    "CRITICAL": "#ff4b4b",
    "HIGH":     "#ffa500",
    "MEDIUM":   "#1f77b4",
}


# ── Synthetic order generator ─────────────────────────────────────────────────

def generate_synthetic_order() -> dict:
    # Builds a realistic order mirroring actual DataCo dataset fields
    # Geography is derived from market so region and city are always consistent
    month      = random.randint(1, 12)
    market     = random.choice(list(MARKET_GEO.keys()))
    region, city, country = random.choice(MARKET_GEO[market])
    department = random.choice(list(DEPARTMENT_PRODUCTS.keys()))
    product_name, price_min, price_max = random.choice(DEPARTMENT_PRODUCTS[department])
    product_price   = round(random.uniform(price_min, price_max), 2)
    discount_rate   = round(random.uniform(0.0, 0.35), 2)
    discount_amount = round(product_price * discount_rate, 2)
    quantity        = random.randint(1, 5)

    # Fields the model scores on
    model_fields = {
        "days_for_shipment_scheduled": random.randint(1, 6),
        "order_month":                 month,
        "order_quarter":               (month - 1) // 3 + 1,
        "region_late_rate":            round(random.uniform(0.48, 0.58), 4),
        "payment_type":                random.choice(PAYMENT_TYPES),
        "customer_segment":            random.choice(CUSTOMER_SEGMENTS),
        "market":                      market,
        "order_status":                random.choice(ORDER_STATUSES),
        "shipping_mode":               random.choice(SHIPPING_MODES),
        "order_item_discount_rate":    discount_rate,
    }

    # Display fields shown in the dialog — not sent to the model
    display_fields = {
        "order_id":            random.randint(70000, 99999),
        "order_region":        region,
        "order_city":          city,
        "order_country":       country,
        "department_name":     department,
        "product_name":        product_name,
        "product_price":       product_price,
        "order_item_discount": discount_amount,
        "order_item_quantity": quantity,
        "order_date":          datetime.now().strftime("%m/%d/%Y %H:%M"),
    }

    # Merge into one dict — score_order extracts only model_fields keys before posting
    return {**model_fields, **display_fields}


# ── API call ──────────────────────────────────────────────────────────────────

def score_order(order: dict) -> dict | None:
    # Extract only the fields the model expects — display fields are never sent to the API
    model_payload = {
        "days_for_shipment_scheduled": order["days_for_shipment_scheduled"],
        "order_month":                 order["order_month"],
        "order_quarter":               order["order_quarter"],
        "region_late_rate":            order["region_late_rate"],
        "payment_type":                order["payment_type"],
        "customer_segment":            order["customer_segment"],
        "market":                      order["market"],
        "order_status":                order["order_status"],
        "shipping_mode":               order["shipping_mode"],
        "order_item_discount_rate":    order["order_item_discount_rate"],
    }
    try:
        response = requests.post(API_URL, json=model_payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Risk explanation engine ───────────────────────────────────────────────────

def explain_order(order: dict) -> tuple:
    # Returns (top_reason: str, supporting_reasons: list[str])
    # Priority number controls what surfaces as the primary driver vs contributing factor
    # Lower number = higher priority = more operationally intuitive to a supervisor
    # Payment type and discount rate are intentionally deprioritized below operational signals

    all_reasons = []

    if order["days_for_shipment_scheduled"] <= 2:
        all_reasons.append((
            1,
            f"Scheduled shipping window is only {order['days_for_shipment_scheduled']} day(s). "
            "This is the strongest individual predictor of late delivery in historical data.",
        ))

    if order["shipping_mode"] == "Same Day":
        all_reasons.append((
            2,
            "Same Day shipping has the highest late delivery rate across all shipping modes. "
            "The compressed fulfillment window leaves no buffer for delays."
        ))

    if order["order_status"] == "SUSPECTED_FRAUD":
        all_reasons.append((
            3,
            "Order is flagged as Suspected Fraud. These orders have a significantly elevated "
            "late delivery rate due to additional verification holds before shipment."
        ))

    if order["order_status"] in ["PENDING", "PROCESSING"]:
        all_reasons.append((
            4,
            f"Order status is {STATUS_LABELS[order['order_status']]}. "
            "Orders stalled at this stage are more likely to miss their delivery window."
        ))

    if order["market"] == "LATAM":
        all_reasons.append((
            5,
            "LATAM market orders show a higher late delivery rate compared to other regions, "
            "driven by longer transit times and customs variability."
        ))

    if order["payment_type"] != "TRANSFER":
        # Framed as a processing risk, not a comparison to a preferred payment type
        all_reasons.append((
            6,
            f"Payment method is {order['payment_type']}. Non-transfer payments carry a higher risk "
            "of processing delays that can hold up order fulfillment before shipment clears."
        ))

    if order["order_item_discount_rate"] > 0.2:
        # Kept as contributing factor only — intentionally lowest priority
        # Real model signal but not intuitive to surface as a primary driver to ops
        all_reasons.append((
            7,
            f"Discount rate of {order['order_item_discount_rate']:.0%} falls into the aggressive tier. "
            "Heavily discounted orders have historically correlated with fulfillment delays."
        ))

    if not all_reasons:
        return (
            "No dominant risk factor identified.",
            ["The score reflects a combination of moderate signals across multiple features."]
        )

    # Sort by priority — lowest number surfaces as the primary driver
    all_reasons.sort(key=lambda x: x[0])
    top_reason = all_reasons[0][1]
    supporting = [r[1] for r in all_reasons[1:]]

    return top_reason, supporting


# ── Dialog popup ──────────────────────────────────────────────────────────────

@st.dialog("Order Detail and Risk Breakdown", width="large")
def show_order_dialog():
    # Reads selected order index from session state — set by the card View Details button
    idx       = st.session_state.selected_order_index
    row       = st.session_state.risk_queue[idx]
    raw_order = st.session_state.raw_orders[idx]

    top_reason, supporting_reasons = explain_order(raw_order)
    tier = row["Risk Tier"]

    # Primary risk factor banner — color matches tier severity
    if tier == "CRITICAL":
        st.error(f"**Primary Risk Factor:** {top_reason}")
    elif tier == "HIGH":
        st.warning(f"**Primary Risk Factor:** {top_reason}")
    else:
        st.info(f"**Primary Risk Factor:** {top_reason}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Order Details**")
        details = {
            "Order ID":           f"#{raw_order['order_id']}",
            "Order Date":         raw_order["order_date"],
            "Late Delivery Risk": row["Late Delivery Risk"],
            "Risk Tier":          tier,
            "Shipping Mode":      row["Shipping Mode"],
            "Order Status":       row["Order Status"],
            "Days Scheduled":     row["Days Scheduled"],
            "Market":             row["Market"],
            "Order Region":       raw_order["order_region"],
            "Destination":        f"{raw_order['order_city']}, {raw_order['order_country']}",
            "Customer Segment":   raw_order["customer_segment"],
            "Payment Type":       raw_order["payment_type"],
        }
        for label, value in details.items():
            st.write(f"**{label}:** {value}")

    with col_right:
        st.markdown("**Product Details**")
        product_details = {
            "Department":     raw_order["department_name"],
            "Product":        raw_order["product_name"],
            "Unit Price":     f"${raw_order['product_price']:.2f}",
            "Quantity":       raw_order["order_item_quantity"],
            "Discount Rate":  f"{raw_order['order_item_discount_rate']:.0%}",
            "Discount Value": f"${raw_order['order_item_discount']:.2f}",
        }
        for label, value in product_details.items():
            st.write(f"**{label}:** {value}")

        st.divider()

        st.markdown("**Contributing Risk Factors**")
        if supporting_reasons:
            for reason in supporting_reasons:
                st.write(f"- {reason}")
        else:
            st.write("No additional contributing factors beyond the primary risk driver.")


# ── Card renderer ─────────────────────────────────────────────────────────────

def render_order_card(idx: int, row: dict, raw_order: dict):
    # Renders a single order as a bordered card
    # Surface fields are kept lean — full detail lives in the dialog
    tier  = row["Risk Tier"]
    color = TIER_COLORS.get(tier, "#888888")

    with st.container(border=True):
        col_badge, col_score, col_mode, col_status, col_market, col_dest, col_product, col_days, col_btn = st.columns(
            [1.2, 1, 1.4, 1.4, 1, 1.4, 1.5, 0.8, 1.4]
        )

        with col_badge:
            # Inline styled span renders a colored tier badge without a full dataframe
            st.markdown(
                f"<span style='background-color:{color};color:white;"
                f"padding:3px 10px;border-radius:4px;font-size:0.8rem;"
                f"font-weight:600;'>{tier}</span>",
                unsafe_allow_html=True,
            )

        with col_score:
            st.markdown(f"**{row['Late Delivery Risk']}**")

        with col_mode:
            st.caption("Shipping Mode")
            st.write(row["Shipping Mode"])

        with col_status:
            st.caption("Order Status")
            st.write(row["Order Status"])

        with col_market:
            st.caption("Market")
            st.write(row["Market"])

        with col_dest:
            st.caption("Destination")
            st.write(f"{raw_order['order_city']}, {raw_order['order_country']}")

        with col_product:
            st.caption("Product")
            st.write(raw_order["product_name"])

        with col_days:
            st.caption("Days")
            st.write(row["Days Scheduled"])

        with col_btn:
            # Stores which order was selected then triggers the dialog
            if st.button("View Details", key=f"detail_btn_{idx}", use_container_width=True):
                st.session_state.selected_order_index = idx
                show_order_dialog()


# ── Main render function ──────────────────────────────────────────────────────

def render_risk_queue():
    st.title("Risk Queue")
    st.write(
        "Orders are scored through the live prediction API. "
        "Only orders flagged as Medium, High, or Critical risk are shown."
    )

    # Initialize session state on first load
    if "risk_queue" not in st.session_state:
        st.session_state.risk_queue = []
    if "raw_orders" not in st.session_state:
        st.session_state.raw_orders = []
    if "selected_order_index" not in st.session_state:
        st.session_state.selected_order_index = None

    # Controls row
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        n_orders = st.slider("Orders to generate", min_value=1, max_value=20, value=5)

    with col2:
        st.write("")
        generate = st.button("Generate and Score Orders", use_container_width=True)

    with col3:
        st.write("")
        if st.button("Clear Queue", use_container_width=True):
            st.session_state.risk_queue           = []
            st.session_state.raw_orders           = []
            st.session_state.selected_order_index = None
            st.rerun()

    # Generate, score, and store at-risk orders
    if generate:
        with st.spinner(f"Scoring {n_orders} orders..."):
            for _ in range(n_orders):
                order  = generate_synthetic_order()
                result = score_order(order)
                if result and result["risk_tier"] in AT_RISK_TIERS:
                    row = {
                        "Time":               datetime.now().strftime("%I:%M %p"),
                        "Late Delivery Risk":  f"{result['risk_score'] * 100:.0f}%",
                        "Risk Tier":          result["risk_tier"],
                        "risk_score_raw":     result["risk_score"],
                        "Shipping Mode":      order["shipping_mode"],
                        "Order Status":       STATUS_LABELS[order["order_status"]],
                        "Market":             order["market"],
                        "Days Scheduled":     order["days_for_shipment_scheduled"],
                        "Customer Segment":   order["customer_segment"],
                        "Payment Type":       order["payment_type"],
                    }
                    st.session_state.risk_queue.append(row)
                    st.session_state.raw_orders.append(order)

    # Render the queue if there are at-risk orders
    if st.session_state.risk_queue:

        # Sort by risk score descending using an index list
        # Sorting indices keeps risk_queue and raw_orders perfectly aligned
        sorted_indices = sorted(
            range(len(st.session_state.risk_queue)),
            key=lambda i: st.session_state.risk_queue[i]["risk_score_raw"],
            reverse=True,
        )

        # Summary metric counts
        tiers = [st.session_state.risk_queue[i]["Risk Tier"] for i in sorted_indices]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Critical", tiers.count("CRITICAL"))
        col_b.metric("High",     tiers.count("HIGH"))
        col_c.metric("Medium",   tiers.count("MEDIUM"))

        st.divider()

        # Column headers aligned to card columns so the layout reads like a table
        h1, h2, h3, h4, h5, h6, h7, h8, h9 = st.columns(
            [1.2, 1, 1.4, 1.4, 1, 1.4, 1.5, 0.8, 1.4]
        )
        h1.caption("Tier")
        h2.caption("Risk Score")
        h3.caption("Shipping Mode")
        h4.caption("Order Status")
        h5.caption("Market")
        h6.caption("Destination")
        h7.caption("Product")
        h8.caption("Days")
        h9.caption("")

        # Render one card per at-risk order in descending risk score order
        for i in sorted_indices:
            render_order_card(i, st.session_state.risk_queue[i], st.session_state.raw_orders[i])

        st.caption(
            f"{len(sorted_indices)} at-risk orders — sorted by risk score descending. "
            "Low risk orders are not shown."
        )

    else:
        st.info("No at-risk orders in the queue. Click Generate and Score Orders to begin.")