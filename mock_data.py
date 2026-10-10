"""
mock_data.py
============
Deterministic fake data for the ShopEasy multi-agent support system.

Everything is in-memory and "now" is frozen (FIXED_NOW) so every tool
returns the same answer on every run — evaluation scores then reflect the
agent's behaviour, not changing data.

Each order is built to exercise one scenario:
  ORD-1042  Shipped electronics        — status / tracking / address change rejected
  ORD-2099  Delivered clothing         — invoice / refund within window
  ORD-7777  Processing food, 30 min old — cancellable (1-hour window)
  ORD-3141  Delivered furniture        — DOUBLE-CHARGED (billing dispute)
  ORD-5005  Processing electronics     — too old to cancel, address change allowed
  ORD-6060  Shipped international      — held at customs
"""

from datetime import datetime

FIXED_NOW = datetime(2026, 5, 10, 12, 0)
CANCELLATION_WINDOW_MINUTES = 60

CUSTOMERS = {
    "priya@example.com": "CUST-001",
    "marcus@example.com": "CUST-002",
    "elena@example.com": "CUST-003",
}

ORDERS = {
    "ORD-1042": {
        "customer_id": "CUST-001", "status": "Shipped", "category": "electronics",
        "items": [("Wireless headphones", 1, 129.99)],
        "placed_at": datetime(2026, 5, 8, 9, 15), "eta": "2026-05-13", "delivered_on": None,
        "carrier": "UPS", "tracking": "1Z999AA10123456784",
        "last_event": "Departed UPS facility, Louisville KY (2026-05-10 06:40)",
    },
    "ORD-2099": {
        "customer_id": "CUST-002", "status": "Delivered", "category": "clothing",
        "items": [("Denim jacket", 1, 89.50)],
        "placed_at": datetime(2026, 5, 3, 14, 2), "eta": "2026-05-08", "delivered_on": "2026-05-08",
        "carrier": "FedEx", "tracking": "7712 3456 9012",
        "last_event": "Delivered, left at front door (2026-05-08 15:21)",
    },
    "ORD-7777": {
        "customer_id": "CUST-001", "status": "Processing", "category": "food",
        "items": [("Gourmet snack box", 1, 45.00)],
        "placed_at": datetime(2026, 5, 10, 11, 30), "eta": "2026-05-15", "delivered_on": None,
        "carrier": None, "tracking": None, "last_event": "Order received, awaiting fulfilment",
    },
    "ORD-3141": {
        "customer_id": "CUST-003", "status": "Delivered", "category": "furniture",
        "items": [("Oak bookshelf", 1, 249.00)],
        "placed_at": datetime(2026, 5, 1, 10, 0), "eta": "2026-05-06", "delivered_on": "2026-05-06",
        "carrier": "XPO Logistics", "tracking": "XPO-558120",
        "last_event": "Delivered, signed by E. Rossi (2026-05-06 11:05)",
    },
    "ORD-5005": {
        "customer_id": "CUST-002", "status": "Processing", "category": "electronics",
        "items": [("USB-C charger", 1, 19.99)],
        "placed_at": datetime(2026, 5, 9, 16, 0), "eta": "2026-05-16", "delivered_on": None,
        "carrier": None, "tracking": None, "last_event": "Picking in warehouse",
    },
    "ORD-6060": {
        "customer_id": "CUST-003", "status": "Shipped", "category": "clothing",
        "items": [("Wool scarf", 1, 35.00)],
        "placed_at": datetime(2026, 4, 25, 8, 30), "eta": "2026-05-12", "delivered_on": None,
        "carrier": "DHL International", "tracking": "JD014600006281",
        "last_event": "Held at customs, Heathrow UK — import duties due from recipient",
    },
}

# Payments per order. ORD-3141 is deliberately charged twice.
PAYMENTS = {
    "ORD-1042": [("CHG-1001", "2026-05-08", 129.99, "Visa ****4242")],
    "ORD-2099": [("CHG-1002", "2026-05-03", 89.50, "Mastercard ****5100")],
    "ORD-7777": [("CHG-1003", "2026-05-10", 45.00, "Visa ****4242")],
    "ORD-3141": [("CHG-1004", "2026-05-01", 249.00, "Amex ****0005"),
                 ("CHG-1005", "2026-05-01", 249.00, "Amex ****0005")],
    "ORD-5005": [("CHG-1006", "2026-05-09", 19.99, "Mastercard ****5100")],
    "ORD-6060": [("CHG-1007", "2026-04-25", 35.00, "Amex ****0005")],
}

# Same numbers as the RAG knowledge base in rag_agent.py (single source of truth).
SHIPPING_OPTIONS = [
    ("Standard", "5-7 business days", "free on orders over $50, otherwise $5.99"),
    ("Express", "1-2 business days", "$15 (orders before 2 PM dispatch same day)"),
    ("International", "10-21 business days", "from $25; customs duties paid by the buyer"),
]

RETURN_WINDOW_DAYS = {"electronics": 15, "clothing": 30, "furniture": 30, "food": 0}
