"""
AusClaim AI — Promptfoo Provider (Module 8)
Connects Promptfoo to the LangGraph claims triage agent for red-teaming evaluation.
Exposes call_api() as required by Promptfoo's Python provider API contract.
See: https://www.promptfoo.dev/docs/providers/python/
"""

import sys
import json

sys.path.append(".")

from src.agents.claims_triage_agent import app


def call_api(prompt, options, context):
    # options and context are required by Promptfoo's Python provider API contract
    # See: https://www.promptfoo.dev/docs/providers/python/
    result = app.invoke({"claim_input": prompt})
    return {"output": json.dumps(result)}
