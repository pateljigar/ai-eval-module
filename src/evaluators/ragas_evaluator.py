"""
AusClaim AI — RAGAS Evaluator (Modules 9)
RAGAS evaluates RAG (knowledge base) context for `research policy` node in the LangGraph claims triage agent.
Evaluates: Context Precision, Context Recall, Answer Relevancy, Faithfulness
Covers 4 claim types: motor_vehicle, property, public_liability, other

Known issue: ragas/llms/base.py requires a manual patch to fix a broken
langchain_community.chat_models.vertexai import on langchain-community>=0.4.
See: https://github.com/vibrantlabsai/ragas/issues/2745
"""

import asyncio

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.evaluators.fixtures.claims import TEST_CLAIMS
from src.agents.claims_triage_agent import app

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=client)

# Create metric
context_precision_scorer = ContextPrecision(llm=llm)
context_recall_scorer = ContextRecall(llm=llm)
relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
faithfulness_scorer = Faithfulness(llm=llm)


def run_agent(claim_input: str) -> dict:
    return app.invoke({"claim_input": claim_input})


AGENT_RESULTS = [run_agent(claim["input"]) for claim in TEST_CLAIMS]


async def test_research_policy_ragas():
    for claim, result in zip(TEST_CLAIMS, AGENT_RESULTS):
        actual = result.get("policy_findings", "")
        question = f"What are the requirements for a {result.get('claim_type')} insurance claim?"
        faithfulness_score = await faithfulness_scorer.ascore(
            user_input=question,
            response=actual,
            retrieved_contexts=[result.get("raw_retrieved_policy_text", "")],
        )
        print(
            f"Faithfulness score for claim '{claim['input']}': {faithfulness_score.value}"
        )

        relevancy_score = await relevancy_scorer.ascore(
            user_input=question, response=actual
        )
        print(
            f"Answer Relevancy score for claim '{claim['input']}': {relevancy_score.value}"
        )

        context_precision_score = await context_precision_scorer.ascore(
            user_input=question,
            retrieved_contexts=[result.get("raw_retrieved_policy_text", "")],
            reference=claim["expected_policy_findings"],
        )
        print(
            f"Context Precision score for claim '{claim['input']}': {context_precision_score.value}"
        )

        context_recall_score = await context_recall_scorer.ascore(
            user_input=question,
            retrieved_contexts=[result.get("raw_retrieved_policy_text", "")],
            reference=claim["expected_policy_findings"],
        )
        print(
            f"Context Recall score for claim '{claim['input']}': {context_recall_score.value}"
        )


if __name__ == "__main__":
    asyncio.run(test_research_policy_ragas())
