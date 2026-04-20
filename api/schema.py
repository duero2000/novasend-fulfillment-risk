# Defines the expected structure of an incoming order scoring request
# Pydantic validates all fields automatically before the model sees any data

from pydantic import BaseModel, Field
from typing import Literal

class OrderInput(BaseModel):

    # Numeric features passed directly to the model without transformation
    days_for_shipment_scheduled: int = Field(
        ..., ge=0, le=30,
        description="Number of days promised to customer at order placement"
    )
    order_month: int = Field(
        ..., ge=1, le=12,
        description="Month the order was placed"
    )
    order_quarter: int = Field(
        ..., ge=1, le=4,
        description="Quarter the order was placed"
    )
    region_late_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Historical late delivery rate for the order region"
    )
    order_item_discount_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Discount rate applied to the order item, used to derive discount tier"
    )
    # Categorical fields. Literal types restrict input to valid training values only
    # This prevents the encoding logic from silently producing all-zero OHE rows
    
    shipping_mode: Literal[
        "Standard Class", "First Class", "Second Class", "Same Day"
    ]

    order_status: Literal[
        "COMPLETE", "PENDING", "PROCESSING", "ON_HOLD",
        "SUSPECTED_FRAUD", "CLOSED", "CANCELED",
        "PENDING_PAYMENT", "PAYMENT_REVIEW"
    ]

    market: Literal[
        "Africa", "Europe", "LATAM", "Pacific Asia", "USCA"
    ]

    customer_segment: Literal[
        "Consumer", "Corporate", "Home Office"
    ]

    payment_type: Literal[
        "DEBIT", "TRANSFER", "CASH", "PAYMENT"
    ]