from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase

model = GPTModel(model="gpt-4o-mini")

metric = AnswerRelevancyMetric(model=model, threshold=0.7)

# This test case is designed to pass as the actual output provides the correct information about lodging a car insurance claim.
test_case_pass = LLMTestCase(
    input="How do I lodge a car insurance claim?",
    actual_output="To lodge a car insurance claim, call our 24/7 claims hotline or submit online through the AusClaim portal with your policy number and incident details.",
    expected_output="Contact AusClaim via phone or online portal with your policy number and incident information to lodge a car insurance claim."
)

# This test case is designed to fail as the actual output does not provide the correct information about lodging a car insurance claim.
test_case_bad = LLMTestCase(
    input="How do I lodge a car insurance claim?",
    actual_output="To lodge a car insurance claim, You need to visit in person.",
    expected_output="Contact AusClaim via phone or online portal with your policy number and incident information to lodge a car insurance claim."
)


# This test case is designed to be borderline as the actual output provides some information but is not complete or fully accurate.
test_case_boarder_line = LLMTestCase(
    input="How do I lodge a car insurance claim?",
    actual_output="To lodge a car insurance claim, call our 24/7 claims hotline.",
    expected_output="Contact AusClaim via phone or online portal with your policy number and incident information to lodge a car insurance claim."
)

# Evaluate the test cases using the defined metric
evaluate([test_case_pass, test_case_bad, test_case_boarder_line], [metric])