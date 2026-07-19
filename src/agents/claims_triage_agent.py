"""
AusClaim AI — Claims Triage Agent (Module 1)
LangGraph agent: classify_claim → research_policy → summarise_decision
System under test for DeepEval evaluation pipeline.
Instrumented with Langfuse observability (Module 7).
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import json
from dotenv import load_dotenv
from langfuse import get_client

try:
    from src.agents.policy_store import query_policy
except ModuleNotFoundError:
    from policy_store import query_policy

load_dotenv()
langfuse = get_client()

class ClaimsTriageState(TypedDict):
    claim_input: str
    claim_type: Optional[str]
    urgency: Optional[str]
    policy_findings: Optional[str]
    final_decision: Optional[dict]
    error: Optional[str]


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def classify_claim(state: ClaimsTriageState):
    print("Running Classifying claim:")
    with langfuse.start_as_current_observation(
        as_type="generation", name="classify_claim", model=model.model_name
    ) as span:
        span.update(input={"claim_input": state["claim_input"]})
        messages = [
            (
                "system",
                """You are an AusClaim insurance triage assistant.
        Classify the claim and return ONLY valid JSON with two keys:
        - claim_type: one of motor_vehicle, property, public_liability, other
        - urgency: one of low, medium, high, critical
        - low: no injuries, minor property damage, no immediate action needed
        - medium: some property damage, no injuries, repair can wait
        - high: injuries present, medical attention needed, or property damage requiring urgent repair
        - critical: life-threatening injuries or major structural damage

        Return nothing else. No explanation. No markdown. Just the JSON object.""",
            ),
            ("human", state["claim_input"]),
        ]
        response = model.invoke(messages)
        # Extract token usage from response metadata
        usage = response.response_metadata.get("token_usage", {})
        try:
            claim_data = json.loads(response.content)
            print(
                f"  ✓ claim_type: {claim_data.get('claim_type')}, urgency: {claim_data.get('urgency')}"
            )
            span.update(
                output=claim_data,
                usage_details={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
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
    with langfuse.start_as_current_observation(
        as_type="generation", name="research_policy", model=model.model_name
    ) as span:
        span.update(input={"claim_type": state.get("claim_type", "unknown")})
        retrieved_policy = query_policy(state.get("claim_type", "unknown"))
        messages = [
            (
                "system",
                """You are an AusClaim policy research assistant.
    Use only the policy provided to summarise the requirements for this claim.
    Return ONLY valid JSON with one key:
    - policy_findings: a 1-2 sentence summary based on the retrieved policy

    Return nothing else. No markdown. Just the JSON object.""",
            ),
            (
                "human",
                f"Claim type: {state.get('claim_type', 'unknown')}\nPolicy: {retrieved_policy}",
            ),
        ]
        response = model.invoke(messages)
        # Extract token usage from response metadata
        usage = response.response_metadata.get("token_usage", {})
        try:
            claim_data = json.loads(response.content)
            print(f"  ✓ policy_findings: {claim_data.get('policy_findings')}")
            span.update(
                output=claim_data,
                usage_details={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            )
            # FAILURE MODE REVERTED — was: return {"claim_type": "CORRUPTED"}
            return {"policy_findings": claim_data.get("policy_findings")}
        except json.JSONDecodeError:
            return {"error": "Failed to research policy"}


def summarise_decision(state: ClaimsTriageState):
    print("Running Summarize decision:")
    with langfuse.start_as_current_observation(
        as_type="generation", name="summarise_decision", model=model.model_name
    ) as span:
        span.update(
            input={
                "claim_type": state.get("claim_type", "unknown"),
                "urgency": state.get("urgency", "unknown"),
                "policy_findings": state.get("policy_findings", "not available"),
            }
        )
        messages = [
            (
                "system",
                """You are an AusClaim senior claims assessor.

You will receive a claim type, urgency level, and policy findings from a prior research step.
Your job is to synthesise that information and produce a final decision.
Your recommendation must be a complete, properly capitalised sentence.
Do not assert facts about damage amounts unless explicitly stated in the claim input.
Only use conditional language when referencing policy thresholds.

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
                f"Claim type: {state.get('claim_type', 'unknown')}, Urgency: {state.get('urgency', 'unknown')}, Policy findings: {state.get('policy_findings', 'not available')}",
            ),
        ]
        response = model.invoke(messages)
        # Extract token usage from response metadata
        usage = response.response_metadata.get("token_usage", {})
        try:
            claim_data = json.loads(response.content)
            span.update(
                output=claim_data,
                usage_details={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            )
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

if __name__ == "__main__":
    result = app.invoke({"claim_input": "Car accident on M1, airbags deployed"})
    print(result)
