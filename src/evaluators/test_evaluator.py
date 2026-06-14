class TestEvaluator:
    
    def __init__(self, threshold, name):
        self.threshold = threshold
        self.name = name

    def evaluateAnswer(self, score):
        if score > self.threshold:
            return 1
        else:
            return 0
    
test_evaluator = TestEvaluator(threshold=0.7, name="relevancy")
print(test_evaluator.evaluateAnswer(score=0.9))  # Output: 1
