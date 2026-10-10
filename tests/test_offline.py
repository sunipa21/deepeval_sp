"""
Offline checks — no API keys, no tokens. Run: pytest tests/ -q

Covers the deterministic parts of the system: tool stubs, routing metrics,
scenario integrity, and supervisor -> specialist -> tool orchestration
(including fault injection) driven by a scripted fake LLM.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evals")]

import pytest
from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import support_tools as t
from multi_agent import AGENT_NAMES, FAULTS, build_support_system
from routing_metrics import AgentRoutingMetric, ToolSelectionMetric
from scenarios import ALL_SCENARIOS, EXTENDED_SCENARIOS, FAULT_SCENARIOS, SCENARIOS


# ---------------------------------------------------------------------------
# Tool stubs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tool, args, expected", [
    (t.get_order_status, {"order_id": "ord-1042"}, "Shipped"),
    (t.get_order_status, {"order_id": "ORD-9999"}, "No order found"),
    (t.cancel_order, {"order_id": "ORD-7777"}, "cancelled"),
    (t.cancel_order, {"order_id": "ORD-5005"}, "cannot be cancelled"),
    (t.get_invoice, {"order_id": "ORD-2099"}, "total $96.66"),
    (t.get_payment_history, {"order_id": "ORD-3141"}, "DUPLICATE CHARGE DETECTED"),
    (t.request_refund, {"order_id": "ORD-3141", "reason": "duplicate charge"}, "RF-3141-D approved"),
    (t.request_refund, {"order_id": "ORD-1042", "reason": "return"}, "not been delivered"),
    (t.request_refund, {"order_id": "ORD-2099", "reason": "return"}, "Return RF-2099 opened"),
    (t.track_shipment, {"order_id": "ORD-6060"}, "Held at customs"),
    (t.track_shipment, {"order_id": "ORD-7777"}, "has not shipped"),
    (t.update_shipping_address, {"order_id": "ORD-1042", "new_address": "x"}, "cannot be changed"),
    (t.update_shipping_address, {"order_id": "ORD-5005", "new_address": "22 Baker St"}, "updated"),
    (t.find_orders_by_email, {"email": "priya@example.com"}, "ORD-1042"),
])
def test_tool_stub(tool, args, expected):
    first, second = tool.invoke(args), tool.invoke(args)
    assert expected in first and first == second  # correct and deterministic


def test_call_log_records_calls():
    log = t.start_call_log()
    t.get_shipping_options.invoke({})
    assert [e["name"] for e in log] == ["get_shipping_options"]


# ---------------------------------------------------------------------------
# Routing metrics
# ---------------------------------------------------------------------------
def _case(*names):
    return LLMTestCase(input="q", actual_output="a", tools_called=[ToolCall(name=n) for n in names] or None)


def test_routing_metric():
    exp = {"expected_agents": ["billing_agent"]}
    assert AgentRoutingMetric(expectations=exp).measure(_case("billing_agent", "get_invoice")) == 1.0
    assert AgentRoutingMetric(expectations=exp).measure(_case("shipping_agent")) == 0.0
    assert AgentRoutingMetric(expectations={"expected_agents": []}).measure(_case()) == 1.0


def test_tool_selection_metric():
    exp = {"expected_tools": ["get_invoice"], "forbidden_tools": ["track_shipment"]}
    assert ToolSelectionMetric(expectations=exp).measure(_case("billing_agent", "get_invoice")) == 1.0
    assert ToolSelectionMetric(expectations=exp).measure(_case("get_invoice", "track_shipment")) == 0.0
    m = ToolSelectionMetric(expectations={"expected_tools": []})
    assert m.measure(_case("get_order_status")) == 0.0 and not m.is_successful()


# ---------------------------------------------------------------------------
# Scenario integrity
# ---------------------------------------------------------------------------
KNOWN_TOOLS = {x.name for x in t.ORDER_TOOLS + t.BILLING_TOOLS + t.SHIPPING_TOOLS} | {"search_policies"}


def test_scenarios_are_valid():
    all_s = ALL_SCENARIOS + FAULT_SCENARIOS
    assert len(SCENARIOS) >= 20 and len(EXTENDED_SCENARIOS) >= 10
    assert len({s.id for s in all_s}) == len(all_s)
    for s in all_s:
        assert set(s.expected_agents) <= set(AGENT_NAMES) | {"escalate_to_human"}, s.id
        assert set(s.expected_tools + s.forbidden_tools) <= KNOWN_TOOLS, s.id
        assert s.fault is None or s.fault in FAULTS, s.id
        assert s.to_golden().additional_metadata["scenario_id"] == s.id


# ---------------------------------------------------------------------------
# Orchestration with a scripted fake LLM
# ---------------------------------------------------------------------------
class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


def _factory(scripts):
    return lambda role: ScriptedModel(messages=iter(scripts.get(role, [AIMessage(content="n/a")])))


def _run(system, text):
    log = t.start_call_log()
    reply = system.invoke({"messages": [{"role": "user", "content": text}]})["messages"][-1].text
    return reply, [e["name"] for e in log]


def test_supervisor_delegates_to_billing():
    system = build_support_system(model_factory=_factory({
        "supervisor": [_call("billing_agent", request="invoice ORD-2099"), AIMessage(content="Here it is.")],
        "billing_agent": [_call("get_invoice", order_id="ORD-2099"), AIMessage(content="INV-2099")],
    }))
    reply, calls = _run(system, "invoice for ORD-2099")
    assert calls == ["billing_agent", "get_invoice"] and reply == "Here it is."


def test_fault_misroute_billing_reaches_shipping_specialist():
    system = build_support_system(fault="misroute_billing", model_factory=_factory({
        "supervisor": [_call("billing_agent", request="invoice ORD-2099"), AIMessage(content="done")],
        "shipping_agent": [_call("track_shipment", order_id="ORD-2099"), AIMessage(content="tracking")],
    }))
    _, calls = _run(system, "invoice for ORD-2099")
    assert calls == ["billing_agent", "track_shipment"]  # billing request answered by shipping tools


def test_fault_missing_cancel_tool_blocks_cancellation():
    system = build_support_system(fault="missing_cancel_tool", model_factory=_factory({
        "supervisor": [_call("order_agent", request="cancel ORD-7777"), AIMessage(content="sorry")],
        "order_agent": [_call("cancel_order", order_id="ORD-7777"), AIMessage(content="could not cancel")],
    }))
    _, calls = _run(system, "cancel ORD-7777")
    assert calls == ["order_agent"]  # cancel_order was requested but is not registered
