import unittest

from llm_judge.show_result import calculate_average_score, calculate_win_rate


class TestCalculateAverageScore(unittest.TestCase):
    def test_calculate_average_score(self):
        results = [{"score": 1}, {"score": 2}, {"score": 3}]
        self.assertEqual(calculate_average_score(results), 2)

        results = [{"score": 1}, {"score": 2}, {"score": 3}, {"score": 4}]
        self.assertEqual(calculate_average_score(results), 2.5)


class TestCalculateWinRate(unittest.TestCase):
    def test_calculate_win_rate(self):
        results = [
            {"g1_winner": "model_1", "g2_winner": "model_1"},
        ]
        self.assertEqual(
            calculate_win_rate(results),
            {
                "model_1": {"win_rate": 1.0, "adjusted_win_rate": 1.0},
                "model_2": {"win_rate": 0.0, "adjusted_win_rate": 0.0},
            },
        )

        results = [
            {"g1_winner": "model_1", "g2_winner": "model_2"},
        ]
        self.assertEqual(
            calculate_win_rate(results),
            {
                "model_1": {"win_rate": 0.0, "adjusted_win_rate": 0.5},
                "model_2": {"win_rate": 0.0, "adjusted_win_rate": 0.5},
            },
        )

        results = [
            {"g1_winner": "model_1", "g2_winner": "tie"},
        ]
        self.assertEqual(
            calculate_win_rate(results),
            {
                "model_1": {"win_rate": 0.0, "adjusted_win_rate": 0.5},
                "model_2": {"win_rate": 0.0, "adjusted_win_rate": 0.5},
            },
        )
