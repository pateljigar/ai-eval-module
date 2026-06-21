"""
AusClaim AI — Claims Triage Agent (Module 1)
LangGraph agent: classify_claim → research_policy → summarise_decision
System under test for Module 2 DeepEval evaluation pipeline.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import json
from dotenv import load_dotenv

load_dotenv()


class ClaimsTriageState(TypedDict):
    claim_input: str
    claim_type: Optional[str]
    urgency: Optional[str]
    policy_findings: Optional[str]
    final_decision: Optional[dict]
    error: Optional[str]


model = ChatOpenAI(model="gpt-4o-mini")


def classify_claim(state: ClaimsTriageState):
    print("Running Classifying claim:")
    messages = [
        (
            "system",
            """You are an AusClaim insurance triage assistant.
Classify the claim and return ONLY valid JSON with two keys:
- claim_type: one of motor_vehicle, property, public_liability, other
- urgency: one of low, medium, high, critical

Return nothing else. No explanation. No markdown. Just the JSON object.""",
        ),
        ("human", state["claim_input"]),
    ]
    response = model.invoke(messages)
    try:
        claim_data = json.loads(response.content)
        print(
            f"  ✓ claim_type: {claim_data.get('claim_type')}, urgency: {claim_data.get('urgency')}"
        )
        return {
            "claim_type": claim_data.get("claim_type"),
            "urgency": claim_data.get("urgency"),
        }
    except json.JSONDecodeError:
        return {"error": "Failed to parse claim classification"}


# DELIBERATE FAILURE MODE — silent state corruption
# research_policy writes to claim_type instead of policy_findings, silently overwriting
# classify_claim's correct output. LangGraph has no built-in assertion layer — every node
# can write to any key without restriction, and the graph completes successfully with no
# error raised. The recommendation step receives corrupted data and produces a wrong
# decision with full confidence. This is a key risk in multi-step LLM agents.
# Module 2 DeepEval evaluator will detect this by asserting:
# - claim_type is one of: motor_vehicle, property, public_liability, other
# - policy_findings is present and non-empty in final state
# - final_decision.claim_type matches the original classification output
def research_policy(state: ClaimsTriageState):
    print("Running Research policy:")
    messages = [
        (
            "system",
            """You are an AusClaim policy research assistant.

You have access to the following policy rules:
- Motor vehicle claims require a police report if damage exceeds $2500
- Standard excess is $650
- Coverage includes third-party property damage
- Public liability claims require an incident report and witness statements within 14 days
- Property claims require photos and repair quotes within 30 days
- Critical urgency claims are escalated to a senior assessor within 2 hours

Given the claim type, return ONLY valid JSON with one key:
- policy_findings: a 1-2 sentence summary of the relevant policy rules for this claim type

Return nothing else. No markdown. Just the JSON object.""",
        ),
        ("human", f"Claim type: {state['claim_type']}"),
    ]
    response = model.invoke(messages)
    try:
        claim_data = json.loads(response.content)
        print(f"  ✓ policy_findings: {claim_data.get('policy_findings')}")
        # FAILURE MODE REVERTED — was: return {"claim_type": "CORRUPTED"}
        return {"policy_findings": claim_data.get("policy_findings")}
    except json.JSONDecodeError:
        return {"error": "Failed to research policy"}


def summarise_decision(state: ClaimsTriageState):
    print("Running Summarize decision:")
    messages = [
        (
            "system",
            """You are an AusClaim senior claims assessor.

You will receive a claim type, urgency level, and policy findings from a prior research step.
Your job is to synthesise that information and produce a final decision.

Return ONLY valid JSON with this exact structure:
{
  "final_decision": {
    "claim_type": "...",
    "urgency": "...",
    "recommendation": "..."
  }
}

Return nothing else. No markdown. Just the JSON object.""",
        ),
        (
            "human",
            f"Claim type: {state['claim_type']}, Urgency: {state['urgency']}, Policy findings: {state.get('policy_findings')}",
        ),
    ]
    response = model.invoke(messages)
    try:
        claim_data = json.loads(response.content)
        print(f"  ✓ final_decision: {claim_data.get('final_decision')}")
        return {"final_decision": claim_data.get("final_decision")}
    except json.JSONDecodeError:
        return {"error": "Failed to parse final decision"}


graph = StateGraph(ClaimsTriageState)

# Add nodes
graph.add_node(classify_claim)
graph.add_node(research_policy)
graph.add_node(summarise_decision)

# Add edges
graph.add_edge(START, "classify_claim")
graph.add_edge("classify_claim", "research_policy")
graph.add_edge("research_policy", "summarise_decision")
graph.add_edge("summarise_decision", END)

app = graph.compile()

result = app.invoke({"claim_input": "Car accident on M1, airbags deployed"})
print(result)
