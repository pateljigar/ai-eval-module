from deepeval import evaluate
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import AnswerRelevancyMetric
from openai import OpenAI

# OpenAI client for generating live chatbot responses
client = OpenAI()

# GPT-4o-mini as the LLM judge for evaluation
model = GPTModel(model="gpt-4o-mini")

# Faithfulness checks if the response contradicts the retrieval context (hallucination detection)
faithfulness_metric = FaithfulnessMetric(
    model=model, threshold=0.7, include_reason=True, async_mode=False
)

# Relevancy checks if the response directly addresses the question asked (relevance detection)
relevance_metric = AnswerRelevancyMetric(
    model=model, threshold=0.7, include_reason=True, async_mode=False
)

# Simulated RAG knowledge base — source of truth for the chatbot
rag_knowledge_base = [
    "AusClaim covers windscreen repair but not full replacement under the basic plan.",
    "Claims can be lodged via phone or the online portal with a valid policy number.",
]

# Test questions covering claim lodging and coverage scenarios
questions = [
    "How do I lodge a car insurance claim?",
    "Does Ausclaim cover windscreen coverage?",
]


# Calls GPT-4o-mini as the chatbot, constrained to retrieval context only
def generate_response(question, retrieval_context):
    messages = [
        {
            "role": "system",
            "content": f"You are an AusClaim assistant. Only answer based on this context: {retrieval_context}",
        },
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


# Test case: live LLM response evaluated against retrieval context
test_case_claim_lodging = LLMTestCase(
    input=questions[0],
    actual_output=generate_response(
        question=questions[0], retrieval_context=rag_knowledge_base
    ),
    retrieval_context=rag_knowledge_base,
)

test_case_windscreen_coverage = LLMTestCase(
    input=questions[1],
    actual_output=generate_response(
        question=questions[1], retrieval_context=rag_knowledge_base
    ),
    retrieval_context=rag_knowledge_base,
)

# Run faithfulness and relevancy evaluation sequentially
for test_case in [test_case_claim_lodging, test_case_windscreen_coverage]:
    print(f"Question: {test_case.input}")
    print(f"Generated Response: {test_case.actual_output}")
    print("---")
    evaluate([test_case], [faithfulness_metric, relevance_metric])
