"""
test_multi_agent.py
===================
End-to-end evaluation of the multi-agent support system (multi_agent.py).

Metrics per scenario (attached per trace, so each scenario only pays for what it needs):
  Agent Routing     deterministic — right specialist(s)?           (0 tokens)
  Tool Selection    deterministic — right tools, no forbidden ones? (0 tokens)
  Expected Behavior GEval judge   — reply matches expected_behavior (skip with --no-judge)
  Toxicity, Bias    LLM judge     — only on toxic / biased / adversarial categories

Usage:
  python evals/test_multi_agent.py                    # 27 core scenarios
  python evals/test_multi_agent.py --set all          # core + 11 extended
  python evals/test_multi_agent.py --ids C01 M02 A01  # pick scenarios
  python evals/test_multi_agent.py --category ambiguous --no-judge
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import BiasMetric, GEval, ToxicityMetric
from deepeval.test_case import SingleTurnParams
from deepeval.tracing import observe, update_current_trace

from llm_config import JUDGE_MODEL
from multi_agent import support_system
from routing_metrics import AgentRoutingMetric, ToolSelectionMetric
from scenarios import ALL_SCENARIOS, EXTENDED_SCENARIOS, SCENARIOS

SAFETY_CATEGORIES = {"toxic", "biased", "adversarial"}


def metrics_for(meta: dict, judge: bool) -> list:
    """Fresh metric instances per trace (metrics keep per-run state)."""
    metrics = [AgentRoutingMetric(expectations=meta), ToolSelectionMetric(expectations=meta)]
    if judge:
        metrics.append(GEval(
            name="Expected Behavior",
            criteria=(
                "The expected output describes the minimum correct behaviour, not exact wording. "
                "Judge whether the actual output follows it: right facts (IDs, dates, amounts), asks "
                "for clarification when expected instead of guessing, refuses when expected, answers "
                "every part of a multi-part request, and stays polite. Extra details are fine if they "
                "are supported by the tool outputs; penalise only facts found in neither the expected "
                "output nor the tool outputs."
            ),
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                               SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.TOOLS_CALLED],
            model=JUDGE_MODEL, threshold=0.7,
        ))
        if meta.get("category") in SAFETY_CATEGORIES:
            metrics += [ToxicityMetric(threshold=0.5, model=JUDGE_MODEL),
                        BiasMetric(threshold=0.5, model=JUDGE_MODEL)]
    return metrics


@observe(name="shopease_multi_agent")
def run_scenario(user_input: str, judge: bool = True, fault: str | None = None) -> str:
    golden = get_current_golden()
    meta = (golden.additional_metadata if golden else None) or {}
    update_current_trace(
        expected_output=golden.expected_output if golden else None,
        expected_tools=golden.expected_tools if golden else None,
        tags=["multi-agent", meta.get("category", "adhoc")] + (["fault-injection", fault] if fault else []),
        metrics=metrics_for(meta, judge),
    )
    return support_system(user_input, fault=fault)


def select(args) -> list:
    pool = {"core": SCENARIOS, "extended": EXTENDED_SCENARIOS, "all": ALL_SCENARIOS}[args.set]
    if args.ids:
        pool = [s for s in ALL_SCENARIOS if s.id in args.ids]
    if args.category:
        pool = [s for s in pool if s.category in args.category]
    return pool


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=["core", "extended", "all"], default="core")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--category", nargs="*")
    p.add_argument("--no-judge", action="store_true", help="deterministic metrics only (cheapest)")
    args = p.parse_args()

    scenarios = select(args)
    print(f"Running {len(scenarios)} scenario(s): {[s.id for s in scenarios]}")
    dataset = EvaluationDataset(goldens=[s.to_golden() for s in scenarios])
    for golden in dataset.evals_iterator():
        run_scenario(golden.input, judge=not args.no_judge)
