"""
support_tools.py
================
Deterministic tool stubs for the multi-agent support system, grouped by the
specialist agent that owns them. All data comes from mock_data.py — no real
integrations, no LLM calls.

Every tool call is appended to a per-request call log (see `record` /
`start_call_log`). multi_agent.py copies that log into the DeepEval trace as
`tools_called`, because DeepEval only auto-collects tool spans that are
*direct* children of a span — tools called inside a sub-agent would otherwise
be invisible to ToolCorrectness-style metrics.
"""

import zlib
from contextvars import ContextVar
from datetime import timedelta

from langchain_core.tools import tool

from mock_data import (
    CANCELLATION_WINDOW_MINUTES, CUSTOMERS, FIXED_NOW, ORDERS, PAYMENTS,
    RETURN_WINDOW_DAYS, SHIPPING_OPTIONS,
)

# ---------------------------------------------------------------------------
# Call log — one list per request, shared by supervisor and specialists.
# ---------------------------------------------------------------------------
_call_log: ContextVar[list | None] = ContextVar("call_log", default=None)


def start_call_log() -> list:
    log: list = []
    _call_log.set(log)
    return log


def current_call_log() -> list | None:
    return _call_log.get()


def record(name: str, args: dict, output: str) -> str:
    log = _call_log.get()
    if log is not None:
        log.append({"name": name, "args": args, "output": output})
    return output


def _order(order_id: str):
    return ORDERS.get(order_id.strip().upper())


def _not_found(order_id: str) -> str:
    return f"No order found with ID {order_id}. Ask the customer to double-check the ID."


def _cancellable(order: dict) -> bool:
    age = FIXED_NOW - order["placed_at"]
    return order["status"] == "Processing" and age <= timedelta(minutes=CANCELLATION_WINDOW_MINUTES)


# ---------------------------------------------------------------------------
# Order agent tools
# ---------------------------------------------------------------------------
@tool
def get_order_status(order_id: str) -> str:
    """Get status, ETA, items and cancellation eligibility of an order by its ID (e.g. ORD-1042)."""
    order = _order(order_id)
    if not order:
        out = _not_found(order_id)
    else:
        items = ", ".join(f"{q} x {n}" for n, q, _ in order["items"])
        out = (f"Order {order_id.upper()}: {order['status']}. Items: {items}. "
               f"Category: {order['category']}. Placed {order['placed_at']:%Y-%m-%d %H:%M}. "
               f"ETA {order['eta']}. Delivered on: {order['delivered_on'] or 'not yet'}. "
               f"Cancellable now: {'yes' if _cancellable(order) else 'no'}.")
    return record("get_order_status", {"order_id": order_id}, out)


@tool
def find_orders_by_email(email: str) -> str:
    """List the order IDs belonging to the customer's own email address."""
    cust = CUSTOMERS.get(email.strip().lower())
    if not cust:
        out = f"No customer account found for {email}."
    else:
        ids = [oid for oid, o in ORDERS.items() if o["customer_id"] == cust]
        out = f"Orders for {email}: " + ", ".join(f"{i} ({ORDERS[i]['status']})" for i in ids)
    return record("find_orders_by_email", {"email": email}, out)


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order. Only possible within 60 minutes of placement while still Processing."""
    order = _order(order_id)
    if not order:
        out = _not_found(order_id)
    elif _cancellable(order):
        out = (f"Order {order_id.upper()} cancelled. A full refund of "
               f"${sum(p * q for _, q, p in order['items']):.2f} goes to the original payment "
               f"method within 5-7 business days. Cancellation ref CXL-{order_id.upper()[4:]}.")
    else:
        out = (f"Order {order_id.upper()} cannot be cancelled: it is {order['status']} and was placed "
               f"more than {CANCELLATION_WINDOW_MINUTES} minutes ago or has left fulfilment.")
    return record("cancel_order", {"order_id": order_id}, out)


# ---------------------------------------------------------------------------
# Billing agent tools
# ---------------------------------------------------------------------------
@tool
def get_invoice(order_id: str) -> str:
    """Get the invoice (line items, tax, total) for an order."""
    order = _order(order_id)
    if not order:
        out = _not_found(order_id)
    else:
        subtotal = sum(p * q for _, q, p in order["items"])
        tax = round(subtotal * 0.08, 2)
        lines = "; ".join(f"{n} x{q} ${p:.2f}" for n, q, p in order["items"])
        out = (f"Invoice INV-{order_id.upper()[4:]} for {order_id.upper()}: {lines}. "
               f"Subtotal ${subtotal:.2f}, tax ${tax:.2f}, total ${subtotal + tax:.2f}. "
               f"A PDF copy has been emailed to the account address.")
    return record("get_invoice", {"order_id": order_id}, out)


@tool
def get_payment_history(order_id: str) -> str:
    """List all card charges for an order. Use to check for duplicate or incorrect charges."""
    charges = PAYMENTS.get(order_id.strip().upper())
    if not charges:
        out = _not_found(order_id)
    else:
        lines = "; ".join(f"{cid} {d} ${amt:.2f} on {card}" for cid, d, amt, card in charges)
        dup = len(charges) > 1 and len({c[2] for c in charges}) == 1
        out = f"Charges for {order_id.upper()}: {lines}." + (" DUPLICATE CHARGE DETECTED." if dup else "")
    return record("get_payment_history", {"order_id": order_id}, out)


@tool
def request_refund(order_id: str, reason: str) -> str:
    """Open a refund to the ORIGINAL payment method. reason: e.g. 'duplicate charge', 'return', 'damaged'."""
    oid = order_id.strip().upper()
    order, charges = ORDERS.get(oid), PAYMENTS.get(oid, [])
    r = reason.lower()
    if not order:
        out = _not_found(order_id)
    elif "duplicate" in r:
        if len(charges) > 1:
            out = (f"Refund RF-{oid[4:]}-D approved: duplicate charge {charges[-1][0]} of "
                   f"${charges[-1][2]:.2f} returned to {charges[-1][3]} within 3-5 business days.")
        else:
            out = f"No duplicate charge found on {oid}; refund not opened."
    elif order["status"] != "Delivered":
        out = f"{oid} has not been delivered yet ({order['status']}). Suggest cancellation instead."
    elif "damage" in r or "defect" in r or "broken" in r:
        out = f"Refund RF-{oid[4:]} approved for damaged item: full refund within 5-7 business days."
    elif RETURN_WINDOW_DAYS[order["category"]] == 0:
        out = f"Refund denied: {order['category']} items are non-returnable unless damaged."
    else:
        out = (f"Return RF-{oid[4:]} opened. Ship the item back within "
               f"{RETURN_WINDOW_DAYS[order['category']]} days of delivery; refund issued 5-7 business "
               f"days after we receive it.")
    return record("request_refund", {"order_id": order_id, "reason": reason}, out)


# ---------------------------------------------------------------------------
# Shipping agent tools
# ---------------------------------------------------------------------------
@tool
def track_shipment(order_id: str) -> str:
    """Get carrier, tracking number and latest tracking event for a shipped order."""
    order = _order(order_id)
    if not order:
        out = _not_found(order_id)
    elif not order["tracking"]:
        out = f"{order_id.upper()} has not shipped yet ({order['status']}); no tracking number."
    else:
        out = (f"{order_id.upper()} via {order['carrier']}, tracking {order['tracking']}. "
               f"Latest: {order['last_event']}. ETA {order['eta']}.")
    return record("track_shipment", {"order_id": order_id}, out)


@tool
def get_shipping_options() -> str:
    """List available shipping methods with delivery times and prices."""
    out = "; ".join(f"{n}: {t}, {c}" for n, t, c in SHIPPING_OPTIONS)
    return record("get_shipping_options", {}, out)


@tool
def update_shipping_address(order_id: str, new_address: str) -> str:
    """Change the delivery address. Only possible while the order is still Processing."""
    order = _order(order_id)
    if not order:
        out = _not_found(order_id)
    elif order["status"] != "Processing":
        out = f"Address cannot be changed: {order_id.upper()} is already {order['status']}."
    else:
        out = f"Delivery address for {order_id.upper()} updated to: {new_address}."
    return record("update_shipping_address", {"order_id": order_id, "new_address": new_address}, out)


# ---------------------------------------------------------------------------
# Supervisor-level tool
# ---------------------------------------------------------------------------
@tool
def escalate_to_human(reason: str) -> str:
    """Hand the conversation to a human agent. Use when the customer asks for a human or threatens legal action."""
    out = f"Escalated to a human agent (ticket ESC-{zlib.crc32(reason.encode()) % 10000:04d}). Expected reply within 4 business hours."
    return record("escalate_to_human", {"reason": reason}, out)


ORDER_TOOLS = [get_order_status, find_orders_by_email, cancel_order]
BILLING_TOOLS = [get_invoice, get_payment_history, request_refund]
SHIPPING_TOOLS = [track_shipment, get_shipping_options, update_shipping_address]
