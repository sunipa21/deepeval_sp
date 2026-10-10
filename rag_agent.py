"""
rag_agent.py
============
A customer-support agent that answers questions by retrieving from a small
in-memory policy knowledge base instead of calling fixed lookup functions.

Architecture:
  - Knowledge base: 9 policy documents stored as embeddings (OpenAI)
  - Vector store: langchain_core InMemoryVectorStore (no external DB needed)
  - Tool: search_policies(query) — semantic search, returns top-3 chunks
  - LLM: GPT-4o (same as agent_plain.py, just with retrieval instead of hard-coded data)
  - Instrumentation: DeepEval CallbackHandler + @observe, same 4-line pattern

What's new vs agent_instrumented.py:
  - search_policies tool replaces get_order_status / get_refund_policy
  - Retrieved chunks are captured into the DeepEval trace as retrieval_context
    so that RAG metrics (Faithfulness, Contextual Precision/Recall) can run
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_agent

from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# Knowledge base — product and shipping policies as plain-text documents.
# ---------------------------------------------------------------------------
POLICY_DOCS = [
    # Refund policies
    "Electronics refund policy: Electronics can be returned within 15 days of delivery, "
    "provided the item is unopened and in its original packaging. Opened electronics are "
    "non-returnable unless faulty. Refunds are processed within 5–7 business days.",

    "Clothing refund policy: Clothing items can be returned within 30 days of delivery "
    "with all original tags still attached. Items must be unworn and unwashed. "
    "Refunds are issued to the original payment method within 3–5 business days.",

    "Food refund policy: Food and perishable items are non-returnable for health and "
    "safety reasons. If your food order arrived damaged or spoiled, contact support "
    "within 24 hours and we will issue a full refund or replacement.",

    "Furniture refund policy: Furniture can be returned within 30 days if unassembled "
    "and in original packaging. Assembled furniture cannot be returned unless defective. "
    "Return shipping costs are the customer's responsibility.",

    "Jewellery refund policy: Jewellery and personalised items are non-returnable unless "
    "received in a damaged or incorrect condition. Please inspect items upon delivery.",

    # Shipping policies
    "Standard shipping policy: Standard shipping takes 5–7 business days. "
    "Orders over $50 qualify for free standard shipping. Tracking information is "
    "emailed once the order ships.",

    "Express shipping policy: Express shipping takes 1–2 business days and costs $15. "
    "Express orders placed before 2 PM are dispatched the same day.",

    "International shipping policy: International orders ship via courier and take "
    "10–21 business days. Customs duties and import taxes are the buyer's responsibility. "
    "We ship to over 50 countries.",

    # Order management
    "Order cancellation policy: Orders can be cancelled within 1 hour of placement "
    "for a full refund. After 1 hour, the order enters fulfilment and cannot be "
    "cancelled. Contact support immediately if you need to cancel.",
]


# ---------------------------------------------------------------------------
# Build the vector store once at import time.
# ---------------------------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_texts(POLICY_DOCS)


# ---------------------------------------------------------------------------
# Retrieval tool — returns top-3 policy chunks for a query.
# Also stashes the retrieved text in a module-level list so the @observe
# wrapper can forward it to the DeepEval trace as retrieval_context.
# ---------------------------------------------------------------------------
_last_retrieved: list[str] = []


@tool
def search_policies(query: str) -> str:
    """Search the customer-support knowledge base for policy information."""
    global _last_retrieved
    docs = vector_store.similarity_search(query, k=3)
    chunks = [doc.page_content for doc in docs]
    _last_retrieved = chunks
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

agent = create_agent(
    model=llm,
    tools=[search_policies],
    system_prompt=(
        "You are a friendly customer-support agent. "
        "Use the search_policies tool to look up refund, shipping, and order policies. "
        "Answer only from the retrieved information. Keep replies concise."
    ),
)

# DeepEval callback handler — captures LangChain spans automatically.
deepeval_callback = CallbackHandler()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def rag_support_agent(user_input: str) -> str:
    """Run the RAG agent and return the final reply."""
    global _last_retrieved
    _last_retrieved = []

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [deepeval_callback]},
    )
    reply = result["messages"][-1].content

    # Set clean reply string and retrieved chunks on the trace.
    update_current_trace(
        output=reply,
        retrieval_context=_last_retrieved if _last_retrieved else None,
    )

    return reply


if __name__ == "__main__":
    print(rag_support_agent("What is the return policy for electronics?"))
