"""
test_invalid_tool_calls.py
==========================
Controlled, reproducible WRONG-tool scenarios for investigating routing and
tool-selection failures in DeepEval / Confident AI traces.

Kept separate from test_multi_agent.py on purpose: these runs are EXPECTED
to fail. Each trace is tagged "fault-injection" + the fault name, and the
trace metadata carries {"fault": ..., "tool_sequence": [...]}, so in the
Confident AI portal you can filter for them and open the trace tree to see
the supervisor -> specialist -> tool path that went wrong.

Faults (multi_agent.FAULTS):
  misroute_billing           code bug   — deterministic, reproduces 100%
  swapped_routing_prompt prompt bug — LLM-driven; reproducible at temperature 0
  missing_cancel_tool        deploy bug — deterministic: expected tool cannot be called

Usage:
  python evals/test_invalid_tool_calls.py              # deterministic metrics only (default, cheap)
  python evals/test_invalid_tool_calls.py --judge      # also run the GEval behaviour judge
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepeval.dataset import EvaluationDataset

from multi_agent import FAULTS
from scenarios import FAULT_SCENARIOS
from support_tools import current_call_log
from test_multi_agent import run_scenario

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--judge", action="store_true")
    args = p.parse_args()

    by_id = {s.id: s for s in FAULT_SCENARIOS}
    observed = []
    dataset = EvaluationDataset(goldens=[s.to_golden() for s in FAULT_SCENARIOS])
    for golden in dataset.evals_iterator():
        fault = golden.additional_metadata["fault"]
        run_scenario(golden.input, judge=args.judge, fault=fault)
        observed.append((by_id[golden.additional_metadata["scenario_id"]], [e["name"] for e in current_call_log()]))

    print("\nFault reproduction summary")
    print(f"{'id':4} {'fault':28} {'expected':40} {'observed':45} reproduced")
    for s, calls in observed:
        expected = set(s.expected_agents + s.expected_tools)
        reproduced = not expected <= set(calls) or bool(set(s.forbidden_tools) & set(calls))
        print(f"{s.id:4} {s.fault:28} {', '.join(s.expected_agents + s.expected_tools):40} "
              f"{', '.join(calls) or '(none)':45} {'YES' if reproduced else 'no'}")
    print("\nFault definitions:")
    for name, desc in FAULTS.items():
        print(f"  {name:28} {desc}")
