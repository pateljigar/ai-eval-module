"""
AusClaim AI — Claims Triage Evaluator (Modules 2-4)
DeepEval GEval test cases for each node in the LangGraph claims triage agent.
Evaluates: classify_claim, research_policy, summarise_decision
Covers 4 claim types: motor_vehicle, property, public_liability, other
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
from src.evaluators.fixtures.claims import TEST_CLAIMS
from src.agents.claims_triage_agent import app

# GPT-4o-mini as the LLM judge for DeepEval metrics
model = GPTModel(model="gpt-4o-mini", temperature=0)

# Agent runs once at module level — result shared across all test functions
def run_agent(claim_input: str) -> dict:
    return app.invoke({"claim_input": claim_input})


AGENT_RESULTS = [run_agent(claim["input"]) for claim in TEST_CLAIMS]


def test_classify_claim():
    test_cases = []
    for claim, result in zip(TEST_CLAIMS, AGENT_RESULTS):
        actual = json.dumps(
            {"claim_type": result.get("claim_type"), "urgency": result.get("urgency")}
        )
        expected = json.dumps(
            {
                "claim_type": claim["expected_claim_type"],
                "urgency": claim["expected_urgency"],
            }
        )
        test_cases.append(
            LLMTestCase(
                input=claim["input"], actual_output=actual, expected_output=expected
            )
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
    evaluate(test_cases, [classify_metric])


def test_research_policy():
    test_cases = []
    for claim, result in zip(TEST_CLAIMS, AGENT_RESULTS):
        actual = result.get("policy_findings", "")

        expected = claim["expected_policy_findings"]

        test_cases.append(
            LLMTestCase(
                input=result.get("claim_type"),
                actual_output=actual,
                expected_output=expected,
            )
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
    evaluate(test_cases, [research_metric])


def test_summarise_decision():
    test_cases = []
    for claim, result in zip(TEST_CLAIMS, AGENT_RESULTS):
        node_input = json.dumps(
            {
                "claim_type": result.get("claim_type"),
                "urgency": result.get("urgency"),
                "policy_findings": result.get("policy_findings"),
            }
        )

        actual = json.dumps(result.get("final_decision"))

        expected = claim["expected_recommendation"]

        test_cases.append(
            LLMTestCase(
                input=node_input, actual_output=actual, expected_output=expected
            )
        )

    summarise_metric = GEval(
        name="Decision Summary Accuracy",
        criteria="The recommendation should accurately reflect the policy findings and be consistent with the claim type and urgency.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.6,
    )
    evaluate(test_cases, [summarise_metric])


if __name__ == "__main__":
    test_classify_claim()
    test_research_policy()
    test_summarise_decision()
