# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (already gitignored) with:
```
OPENAI_API_KEY=sk-...               # agent LLM + judge LLM (GPT-4o)
CONFIDENT_API_KEY=confident_us_...  # optional: stream traces to Confident AI dashboard
```

## Running

```bash
# Phase 1 — tool-calling support agent
python agent_instrumented.py          # sanity-check the agent alone

python test_agent.py                  # TaskCompletion + ToolCorrectness
python test_agent_extended.py         # AnswerRelevancy + PromptAlignment + StepEfficiency
python test_agent_correctness.py      # GEval correctness against expected_output

# Phase 2 — RAG agent
python rag_agent.py                   # sanity-check the RAG agent alone
python test_rag_agent.py              # AnswerRelevancy + Faithfulness + ContextualPrecision + ContextualRecall

# Phase 3 — multi-turn chatbot
python chatbot.py                     # sanity-check interactive chatbot (type 'quit' to exit)
python test_chatbot.py                # KnowledgeRetention + RoleAdherence + TurnRelevancy + ConversationalGEval

# Safety evals (reuses agent_instrumented.py)
python test_safety.py                 # Bias + Toxicity + PIILeakage
```

## Project structure

### Phase 1 — tool-calling support agent

| File | Purpose |
|---|---|
| `agent_plain.py` | Baseline agent with two tools (`get_order_status`, `get_refund_policy`). No eval code — the "before" state shown to students. |
| `agent_instrumented.py` | Identical to `agent_plain.py` plus 4 lines of DeepEval instrumentation. All test files import from here. |
| `test_agent.py` | `TaskCompletionMetric` (GPT-4o judge) + `ToolCorrectnessMetric` (rule-based name match). Goldens carry `expected_tools`. |
| `test_agent_extended.py` | `AnswerRelevancyMetric`, `PromptAlignmentMetric`, `StepEfficiencyMetric`. Goldens are input-only — no `expected_output` needed. |
| `test_agent_correctness.py` | `GEval` (Correctness). Goldens carry `expected_output`. This is the non-RAG equivalent of expected-output evaluation. |

### Phase 2 — RAG agent

| File | Purpose |
|---|---|
| `rag_agent.py` | Customer-support agent backed by a 9-document in-memory vector store (`InMemoryVectorStore` + OpenAI embeddings). One tool: `search_policies`. Retrieved chunks are forwarded to the DeepEval trace as `retrieval_context`. |
| `test_rag_agent.py` | `AnswerRelevancyMetric`, `FaithfulnessMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`. Goldens carry `expected_output` (required by Precision/Recall). |

### Phase 3 — multi-turn chatbot

| File | Purpose |
|---|---|
| `chatbot.py` | Multi-turn customer-support chatbot using the OpenAI function-calling API directly (no LangChain). Manages its own tool-call loop and returns `(reply, history, tools_called)` per turn. Same two tools as the agent (`get_order_status`, `get_refund_policy`). |
| `test_chatbot.py` | Conversational evaluation using `ConversationalTestCase` + `Turn` objects. Runs the chatbot live to generate assistant turns, then evaluates with `KnowledgeRetentionMetric`, `RoleAdherenceMetric`, `TurnRelevancyMetric`, and `ConversationalGEval`. Uses `evaluate()`, not `evals_iterator()`. |

### Safety evals

| File | Purpose |
|---|---|
| `test_safety.py` | `BiasMetric`, `ToxicityMetric`, `PIILeakageMetric` against `agent_instrumented.py`. Goldens are input-only and target edge cases: demographic framing, angry customer, SSN in input. |

## Architecture

**Agent instrumentation pattern** (same in both `agent_instrumented.py` and `rag_agent.py`):
1. `CallbackHandler()` — captures every LangChain LLM/tool span automatically
2. `config={"callbacks": [deepeval_callback]}` on `.invoke()` — wires it in
3. `@observe(name="...")` on the outer wrapper — creates the top-level trace
4. `update_current_trace(...)` — explicitly sets `output`, `expected_output`, `expected_tools`, and `retrieval_context` from the current golden so DeepEval metrics can read them

**Why `update_current_trace` is needed**: DeepEval 4.0.4 reads `expected_tools`, `expected_output`, and `retrieval_context` from the trace object, not directly from the golden. They must be copied in explicitly inside the `@observe`-wrapped function using `get_current_golden()`.

**Cross-vendor design**: GPT-4o is used as the impartial judge for all LLM-based metrics; the agent itself also runs on GPT-4o (switched from Claude after the Anthropic key was unavailable).

**Confident AI tracing**: adding `CONFIDENT_API_KEY` to `.env` is the only change needed to enable cloud tracing — no code changes required. DeepEval reads it automatically.

**Chatbot evaluation pattern** (`chatbot.py` / `test_chatbot.py`):
- Uses `ConversationalTestCase` + `Turn` objects instead of `Golden`
- `run_conversation()` drives the chatbot live, collecting `Turn(role="user")` and `Turn(role="assistant", tools_called=[...])` pairs
- `ToolCorrectnessMetric` cannot be used with `ConversationalTestCase`; tool quality is assessed via `ConversationalGEval` with `MultiTurnParams.TOOLS_CALLED`
- Uses `evaluate()` (not `evals_iterator()`) since there is no golden to iterate over

## Metric reference

| Metric | Needs `expected_output`? | Needs `retrieval_context`? | LLM judge? |
|---|---|---|---|
| `TaskCompletionMetric` | No | No | Yes |
| `ToolCorrectnessMetric` | No | No | No (name match) |
| `AnswerRelevancyMetric` | No | No | Yes |
| `PromptAlignmentMetric` | No | No | Yes |
| `StepEfficiencyMetric` | No | No | Yes |
| `GEval` (Correctness) | **Yes** | No | Yes |
| `FaithfulnessMetric` | No | **Yes** | Yes |
| `ContextualPrecisionMetric` | **Yes** | **Yes** | Yes |
| `ContextualRecallMetric` | **Yes** | **Yes** | Yes |
| `KnowledgeRetentionMetric` | No | No | Yes |
| `RoleAdherenceMetric` | No | No | Yes |
| `TurnRelevancyMetric` | No | No | Yes |
| `ConversationalGEval` | No | No | Yes |
| `BiasMetric` | No | No | Yes |
| `ToxicityMetric` | No | No | Yes |
| `PIILeakageMetric` | No | No | Yes |

## Known issues / patches

- **DeepEval 4.0.4 `_make_hashable` bug**: `ToolMessage` objects in `tools_called` are unhashable, crashing `ToolCorrectnessMetric`. Patched directly in `.venv/lib/python3.12/site-packages/deepeval/test_case/llm_test_case.py` — the `else` branch now wraps `hash(obj)` in a try/except and falls back to `str(obj)`.
- **`create_react_agent` deprecation**: migrated to `from langchain.agents import create_agent` with `system_prompt=` replacing `prompt=`. Requires the `langchain` base package (added to `requirements.txt`).
