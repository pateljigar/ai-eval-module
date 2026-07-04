"""
AusClaim AI — Claims Triage Evaluator (Module 2)
DeepEval test cases for each node in the LangGraph claims triage agent.
Evaluates: classify_claim, research_policy, summarise_decision
"""

import json
import sys
import os
from deepeval import evaluate
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase
from deepeval.test_case import SingleTurnParams
from deepeval.metrics import GEval

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.agents.claims_triage_agent import app

# GPT-4o-mini as the LLM judge for DeepEval metrics
model = GPTModel(model="gpt-4o-mini")

CLAIM_INPUT = (
    "Car accident on the highway, minor injuries, urgent medical attention needed."
)

# Agent is invoked once per test function — caching will be added in Module 3
def run_agent(claim_input: str) -> dict:
    return app.invoke({"claim_input": claim_input})


def test_classify_claim():
    result = run_agent(CLAIM_INPUT)

    actual = json.dumps(
        {"claim_type": result.get("claim_type"), "urgency": result.get("urgency")}
    )

    expected = json.dumps({"claim_type": "motor_vehicle", "urgency": "high"})

    test_case_classify_claim = LLMTestCase(
        input=CLAIM_INPUT, actual_output=actual, expected_output=expected
    )

    classify_metric = GEval(
        name="Claim Classification Accuracy",
        criteria="The actual output should contain the correct claim_type and urgency based on the claim description.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.7,
    )
    evaluate([test_case_classify_claim], [classify_metric])


def test_research_policy():
    result = run_agent(CLAIM_INPUT)

    actual = result.get("policy_findings", "")

    expected = "Motor vehicle claims require a police report if damage exceeds $2500, with a standard excess and third-party property coverage."

    test_case_research_policy = LLMTestCase(
        input=result.get("claim_type"), actual_output=actual, expected_output=expected
    )

    research_metric = GEval(
        name="Policy Research Accuracy",
        criteria="The actual output should accurately reflect the policy coverage based on the claim description.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.7,
    )
    evaluate([test_case_research_policy], [research_metric])


def test_summarise_decision():
    result = run_agent(CLAIM_INPUT)
    node_input = json.dumps(
        {
            "claim_type": result.get("claim_type"),
            "urgency": result.get("urgency"),
            "policy_findings": result.get("policy_findings"),
        }
    )

    actual = json.dumps(result.get("final_decision"))

    expected = "Motor vehicle claim received. Police report may be required depending on damage amount. Standard excess of $650 applies."

    test_case_summarise_decision = LLMTestCase(
        input=node_input, actual_output=actual, expected_output=expected
    )

    summarise_metric = GEval(
        name="Decision Summary Accuracy",
        criteria="The recommendation must use conditional language for policy thresholds (e.g. 'if damage exceeds $2500'). It must not assert damage amounts as fact unless stated in the original claim input.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.7,
    )
    evaluate([test_case_summarise_decision], [summarise_metric])


if __name__ == "__main__":
    test_classify_claim()
    test_research_policy()
    test_summarise_decision()
