import unittest

from llm_judge.show_result import calculate_average_score


class TestCalculateAverageScore(unittest.TestCase):
    def test_calculate_average_score(self):
        results = [{"score": 1}, {"score": 2}, {"score": 3}]
        self.assertEqual(calculate_average_score(results), 2)

        results = [{"score": 1}, {"score": 2}, {"score": 3}, {"score": 4}]
        self.assertEqual(calculate_average_score(results), 2.5)
