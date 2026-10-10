"""
routing_metrics.py
==================
Two deterministic (no LLM, zero token cost) DeepEval metrics for the
multi-agent system. They read:

  test_case.tools_called          — actual calls, copied into the trace by multi_agent.py
  expectations                    — expected_agents, expected_tools, forbidden_tools,
                                    passed to the constructor (trace-level metrics in
                                    DeepEval 4.0.4 do not receive golden metadata), or
                                    falling back to test_case.metadata

Why not only ToolCorrectnessMetric? Its default score is recall
(expected ∩ called / expected), so calling an extra *wrong* tool does not
lower the score, and it errors when a scenario expects no tool calls at all
(clarification / refusal cases). Routing errors are exactly what we want to
catch, so these metrics also penalise unexpected and forbidden calls.
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

ROUTES = ("order_agent", "billing_agent", "shipping_agent", "policy_agent", "escalate_to_human")


def _called(test_case: LLMTestCase) -> list[str]:
    return [t.name for t in (test_case.tools_called or [])]


class _DeterministicMetric(BaseMetric):
    evaluation_model = "deterministic (no LLM)"

    def __init__(self, threshold: float, expectations: dict | None = None):
        self.threshold = threshold
        self.expectations = expectations
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = False
        self.verbose_mode = False
        self.evaluation_cost = 0

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def _expect(self, test_case: LLMTestCase, key: str) -> set:
        source = self.expectations if self.expectations is not None else (test_case.metadata or {})
        return set(source.get(key) or [])

    def is_successful(self) -> bool:
        return bool(self.success)

    def _finish(self, score: float, reason: str) -> float:
        self.score, self.reason = score, reason
        self.success = score >= self.threshold
        return score


class AgentRoutingMetric(_DeterministicMetric):
    """F1 between expected and actual specialist routes (incl. escalation).

    No routes expected and none taken (clarify / refuse) scores 1.0.
    """

    def __init__(self, threshold: float = 0.8, expectations: dict | None = None):
        super().__init__(threshold, expectations)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected = self._expect(test_case, "expected_agents")
        called = {n for n in _called(test_case) if n in ROUTES}
        if not expected and not called:
            return self._finish(1.0, "Correctly handled without delegating (clarify / refuse / direct answer).")
        hit = expected & called
        score = 2 * len(hit) / (len(expected) + len(called))
        return self._finish(score, (
            f"expected {sorted(expected) or 'no delegation'}, routed to {sorted(called) or 'nothing'}"
            f"; missing {sorted(expected - called)}; unexpected {sorted(called - expected)}"))

    @property
    def __name__(self):
        return "Agent Routing"


class ToolSelectionMetric(_DeterministicMetric):
    """Recall of expected business tools; 0 if any forbidden tool was called.

    If no tools are expected (clarification / refusal), any tool call scores 0.
    """

    def __init__(self, threshold: float = 1.0, expectations: dict | None = None):
        super().__init__(threshold, expectations)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected = self._expect(test_case, "expected_tools")
        forbidden = self._expect(test_case, "forbidden_tools")
        called = [n for n in _called(test_case) if n not in ROUTES]
        bad = sorted(forbidden & set(called))
        if bad:
            return self._finish(0.0, f"FORBIDDEN tool(s) called: {bad}; sequence {called}")
        if not expected:
            return self._finish(1.0 if not called else 0.0,
                                "No tool call expected" + (f", but called {called}" if called else " and none made."))
        hit = expected & set(called)
        extra = sorted(set(called) - expected)
        return self._finish(len(hit) / len(expected), (
            f"expected {sorted(expected)}, called {called}; missing {sorted(expected - hit)}"
            + (f"; extra (not penalised) {extra}" if extra else "")))

    @property
    def __name__(self):
        return "Tool Selection"
