"""
chatbot.py
==========
A multi-turn customer-support chatbot with tool calling, built with GPT-4o.

Tools available (same data as agent_plain.py):
  - get_order_status(order_id)   → real-time shipping status from in-memory DB
  - get_refund_policy(category)  → refund rules per product category

Unlike agent_plain.py (which delegates the tool-call loop to LangGraph),
this chatbot manages the loop manually so the full message history — including
tool calls and tool results — is preserved across turns. This is what makes
multi-turn evaluation possible with DeepEval.

chat() returns:
  - reply        : the final assistant text shown to the user
  - history      : updated message history to pass into the next turn
  - tools_called : list of (tool_name, args, result) for this turn

Usage:
    history = []
    reply, history, tools = chat("Where is order ORD-1042?", history)
    reply, history, tools = chat("What about the refund policy?", history)

How to run standalone:
    python chatbot.py
"""

from dotenv import load_dotenv

load_dotenv()

import json
from openai import OpenAI

client = OpenAI()

# ---------------------------------------------------------------------------
# In-memory data (same as agent_plain.py)
# ---------------------------------------------------------------------------
ORDERS = {
    "ORD-1042": {"status": "Shipped",    "eta": "2026-05-13"},
    "ORD-2099": {"status": "Delivered",  "eta": "2026-05-08"},
    "ORD-7777": {"status": "Processing", "eta": "2026-05-15"},
}

REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 15 days, unopened.",
    "clothing":    "Clothing can be returned within 30 days with tags attached.",
    "food":        "Food items are non-returnable for safety reasons.",
    "furniture":   "Furniture can be returned within 30 days if unassembled.",
}

# ---------------------------------------------------------------------------
# Tool definitions — passed to the OpenAI API
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the shipping status of a customer order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. ORD-1042",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_refund_policy",
            "description": "Return the refund/return policy for a product category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category, e.g. electronics, clothing, food",
                    }
                },
                "required": ["category"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------
def _execute_tool(name: str, args: dict) -> str:
    if name == "get_order_status":
        order_id = args.get("order_id", "").upper()
        order = ORDERS.get(order_id)
        if not order:
            return f"No order found with ID {order_id}."
        return f"Order {order_id} is {order['status']}. ETA: {order['eta']}."

    if name == "get_refund_policy":
        category = args.get("category", "").lower()
        policy = REFUND_POLICIES.get(category)
        if not policy:
            return f"No refund policy on file for '{category}'."
        return policy

    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly and professional customer-support chatbot \
for ShopEasy, an online retail store.

You have access to two tools:
  - get_order_status : use whenever a customer asks about a specific order
  - get_refund_policy: use whenever a customer asks about returns or refunds

Rules:
- Always use the tools when you have the information needed to call them.
- Be concise and polite.
- Never discuss topics outside of ShopEasy customer support.
- Remember everything the customer tells you in the current conversation."""


# ---------------------------------------------------------------------------
# chat() — one user turn, returns reply + updated history + tools used
# ---------------------------------------------------------------------------
def chat(
    user_message: str,
    history: list[dict],
) -> tuple[str, list[dict], list[dict]]:
    """
    Send one user message, handle any tool calls, return the final reply.

    Returns:
        reply        : final assistant text
        history      : updated message list (include in next call)
        tools_called : list of {"name": ..., "args": ..., "result": ...}
                       for all tools the model called this turn
    """
    history = history + [{"role": "user", "content": user_message}]
    tools_called = []

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message

        # No tool call — final answer
        if not msg.tool_calls:
            reply = msg.content
            history = history + [{"role": "assistant", "content": reply}]
            return reply, history, tools_called

        # Execute every tool the model requested, then loop back
        history = history + [msg]          # append the assistant's tool-call message
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, args)
            tools_called.append({
                "name":   tc.function.name,
                "args":   args,
                "result": result,
            })
            history = history + [{
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            }]


# ---------------------------------------------------------------------------
# Standalone interactive demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("ShopEasy Support Chatbot (with tools) — type 'quit' to exit\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        reply, history, tools = chat(user_input, history)
        if tools:
            for t in tools:
                print(f"  [tool] {t['name']}({t['args']}) → {t['result']}")
        print(f"Bot: {reply}\n")
