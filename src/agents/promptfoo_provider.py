"""
AusClaim AI — Promptfoo Red-Team Provider (Module 8)
Python provider bridge between Promptfoo and the LangGraph claims triage agent.
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
