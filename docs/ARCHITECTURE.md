# ShopEasy Support — Agent & RAG Architecture

This repo contains four generations of the same ShopEasy customer-support
assistant. Each one adds a capability, and each is instrumented for DeepEval
tracing and evaluation.

| Generation | File | Pattern | Tools / knowledge |
|---|---|---|---|
| 1. Tool-calling agent | `agent_plain.py` → `agent_instrumented.py` | single LangChain agent | `get_order_status`, `get_refund_policy`, MCP `get_shipping_options` |
| 2. RAG agent | `rag_agent.py` | single agent + retriever tool | `search_policies` over 9 policy docs |
| 3. Multi-turn chatbot | `chatbot.py` | raw OpenAI function-calling loop | same 2 tools, conversation memory |
| 4. **Multi-agent system** | `multi_agent.py` | supervisor + 4 specialist agents | 10 tools across order / billing / shipping / policy (RAG) |

---

## 1. Multi-agent system (`multi_agent.py`)

### 1.1 Topology

```
                     ┌──────────────────────────────────────────────┐
 customer message ─► │ SUPERVISOR (create_agent)                    │
                     │  decides: clarify | refuse | delegate (1..n) │
                     └───┬───────────┬────────────┬───────────┬─────┴──────────┐
                         ▼           ▼            ▼           ▼                ▼
                   order_agent  billing_agent shipping_agent policy_agent  escalate_to_human
                     │             │             │              │            (direct tool)
        get_order_status     get_invoice    track_shipment   search_policies
        find_orders_by_email get_payment_   get_shipping_       │
        cancel_order           history        options           ▼
                             request_refund update_shipping_  InMemoryVectorStore
                                              address          (rag_agent.py, 9 docs,
                                                               text-embedding-3-small, k=3)
                         │           │            │           │
                         └───────────┴─── mock_data.py (deterministic, FIXED_NOW) ┘
```

**Pattern: "agents as tools".** Each specialist is its own `create_agent`
graph with a narrow system prompt and only its own tools. The supervisor
sees each specialist as one tool (`order_agent(request: str)` and so on),
plus `escalate_to_human`. This keeps:

- **Routing explicit.** Picking a specialist is a tool call, so it shows up
  in traces and can be scored like any other tool choice.
- **Tool menus small.** Each LLM call chooses between at most 5 tools, which
  reduces mis-selection.
- **Multi-intent requests natural.** The supervisor can emit several
  specialist calls in one step. LangGraph runs them in parallel, and the
  supervisor merges the answers.

### 1.2 Decision policy (supervisor prompt)

| Situation | Behaviour |
|---|---|
| Clear single intent | Delegate to one specialist with the order ID and full request. |
| Missing info (no order ID or email) | Ask one clarifying question. **No tool calls, no guessed IDs.** |
| Multi-intent | One call per relevant specialist, then answer every part. |
| Toxic but legitimate | Stay polite, don't mirror insults, still solve the issue. |
| Legal threat or "talk to a human" | `escalate_to_human` in addition to solving what it can. |
| Bias or stereotype bait | Decline the stereotype and treat the customer equally. Answer the legitimate part. |
| Adversarial (prompt injection, data exfiltration, policy bypass) | Refuse without calling tools and offer a legitimate alternative. |
| Any answer | Only facts returned by specialists, at most 120 words. |

Specialists get a shared template (`SPECIALIST_PROMPT`): use tools, report
only tool facts, and say plainly when an action is not possible.

### 1.3 Tools and mock data (`support_tools.py`, `mock_data.py`)

All tools are deterministic stubs over in-memory data with a frozen clock
(`FIXED_NOW = 2026-05-10 12:00`), so the same input always gives the same
tool output. Business rules are encoded in the stubs, not the LLM:

| Rule | Enforced in |
|---|---|
| Cancel only within 60 min and while Processing | `cancel_order` |
| Address change only while Processing | `update_shipping_address` |
| Refunds only to the original payment method (no card parameter exists) | `request_refund` |
| Food non-returnable unless damaged; window per category | `request_refund` |
| Duplicate charge detection | `get_payment_history` |

Each order exists to exercise a scenario: ORD-1042 shipped, ORD-2099
delivered, ORD-7777 cancellable, ORD-3141 double-charged, ORD-5005 too old
to cancel, and ORD-6060 held at customs.

**Single source of truth.** Shipping numbers in `mock_data.SHIPPING_OPTIONS`,
`mcp_server.py`, `policies.txt` and the RAG knowledge base now agree
(Standard 5–7 days, free over $50; Express 1–2 days, $15; International
10–21 days).

### 1.4 RAG component

- **Corpus:** 9 policy documents (`rag_agent.POLICY_DOCS`) covering refunds
  per category, shipping tiers, international shipping and cancellation.
- **Index:** `InMemoryVectorStore` with OpenAI `text-embedding-3-small`. It's
  built lazily on first use: a single embedding batch, with no external
  database.
- **Retrieval:** `search_policies(query)` returns the top-3 chunks. It's used
  standalone in `rag_agent.py` and by the policy specialist in
  `multi_agent.py`, which reuses the same vector store.
- **Grounding:** both prompts say to answer only from retrieved text.
  Retrieved chunks are copied into the DeepEval trace as `retrieval_context`,
  which enables the Faithfulness, Contextual Precision and Contextual Recall
  metrics.

### 1.5 LLM configuration (`llm_config.py`)

| Variable | Default | Used by |
|---|---|---|
| `AGENT_MODEL` | `openai:gpt-4o-mini` | `agent_instrumented.py`, `rag_agent.py`, `multi_agent.py` (`init_chat_model`, temperature 0) |
| `JUDGE_MODEL` | `gpt-4o` | All LLM-as-judge metrics in the new evals |
| `chatbot.py` | `gpt-4o` (hard-coded) | Phase 3 chatbot (raw OpenAI SDK) |

Switching providers is a `.env` change only (for example,
`AGENT_MODEL=anthropic:claude-sonnet-4-5`). The judge stays a different and
stronger model than the agent, to limit self-preference bias.

### 1.6 Observability and tracing

Tracing uses the same 4-line pattern as `agent_instrumented.py`:

1. `CallbackHandler()` from `deepeval.integrations.langchain`, passed on
   `.invoke(config={"callbacks": [...]})`. It captures every LLM, tool and
   chain span. Specialist graphs inherit the callbacks through LangChain's
   context propagation, so the trace is a full tree: supervisor → specialist
   → tool.
2. `@observe(...)` on the outer function (in the eval script, or in
   `__main__`) creates the trace.
3. `update_current_trace(...)` in `support_system()` sets:
   - `output`: the final reply.
   - `tools_called`: **every** agent and tool call, in order, from the
     per-request call log. DeepEval 4.0.4 only auto-collects tool spans that
     are *direct* children of a span, so tools called inside specialists
     would otherwise be missing.
   - `retrieval_context`: the RAG chunks.
   - `metadata`: `agents_called`, `tool_sequence` and `fault`.
4. If `CONFIDENT_API_KEY` is set, traces and test runs stream to the
   Confident AI portal automatically.

### 1.7 Fault injection (for evaluation only)

`build_support_system(fault=...)` produces deliberately broken variants.
These are used only by `evals/test_invalid_tool_calls.py`:

| Fault | Layer | What breaks | Reproducibility |
|---|---|---|---|
| `misroute_billing` | Code | `billing_agent` delegates to the shipping specialist, so no invoice or refund tools are available. | Deterministic |
| `swapped_routing_prompt` | Prompt | Billing and shipping responsibilities are swapped in the supervisor prompt and tool descriptions. | LLM-driven; reproduced in 4 of 4 runs at temperature 0 |
| `missing_cancel_tool` | Deployment / config | `cancel_order` isn't registered on the order specialist. | Deterministic |

Swapping *only* the tool descriptions did **not** mislead the supervisor:
the tool names and prompt routing rules won. That's a useful robustness
finding, and it's why the prompt-level fault swaps both.

### 1.8 Request lifecycle (example M01)

```
"Where is ORD-1042 and what is the return policy for electronics?"
 supervisor ─┬─ order_agent("status of ORD-1042")       ─► get_order_status(ORD-1042)
             └─ policy_agent("electronics return policy") ─► search_policies(...) ─► 3 chunks
 supervisor merges both answers ─► reply
 trace.tools_called = [order_agent, policy_agent, get_order_status, search_policies]
```

Cost per request: about 2 supervisor LLM calls plus about 2 calls per
specialist used. Simple requests take about 4 calls and two-intent requests
about 6, on `gpt-4o-mini`.

---

## 2. Other components (unchanged in design)

- **`agent_instrumented.py`.** Single agent with 2 local tools and 1 MCP tool
  (`mcp_server.py` over stdio). Now uses `llm_config` and launches MCP with
  `sys.executable`, so it works even when the venv isn't on `PATH`.
- **`rag_agent.py`.** Single RAG agent, with the model now from `llm_config`.
  `numpy` was added to the requirements because `InMemoryVectorStore` needs
  it.
- **`chatbot.py`.** Multi-turn chatbot with a manual tool loop, evaluated with
  `ConversationalTestCase` (see `evals/test_chatbot*.py`).

## 3. File map (new and changed)

| File | Status | Purpose |
|---|---|---|
| `llm_config.py` | new | Agent and judge model selection |
| `mock_data.py` | new | Deterministic orders, payments, shipping, return windows |
| `support_tools.py` | new | 10 tool stubs plus a per-request call log |
| `multi_agent.py` | new | Supervisor and specialists, tracing, fault injection |
| `evals/scenarios.py` | new | 27 core, 11 extended and 4 fault scenarios |
| `evals/routing_metrics.py` | new | Deterministic Agent Routing and Tool Selection metrics |
| `evals/test_multi_agent.py` | new | Main eval runner |
| `evals/test_invalid_tool_calls.py` | new | Fault-injection eval runner |
| `tests/test_offline.py` | new | 21 offline checks (pytest, no keys) |
| `agent_instrumented.py`, `rag_agent.py` | changed | Shared LLM config; MCP interpreter fix |
| `mcp_server.py`, `policies.txt` | changed | Shipping data aligned with the knowledge base |
| `requirements.txt` | changed | Pinned `deepeval==4.0.4`; added `langchain-mcp-adapters`, `mcp`, `numpy` |
