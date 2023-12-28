import unittest

from llm_judge.common import MatchPair, MatchSingle


class TestMatchSingle(unittest.TestCase):
    def test_get_score(self) -> None:
        judgement = "[[1]]"
        self.assertEqual(MatchSingle.get_score(judgement), 1)

        judgement = "[[2]]"
        self.assertEqual(MatchSingle.get_score(judgement), 2)

        judgement = "[[rating:3]]"
        self.assertEqual(MatchSingle.get_score(judgement), 3)

        judgement = "[[rating: 4]]"
        self.assertEqual(MatchSingle.get_score(judgement), 4)

        judgement = "[[rating: 4.5]]"
        self.assertEqual(MatchSingle.get_score(judgement), -1)

        judgement = "[[rating: Perfect]]"
        self.assertEqual(MatchSingle.get_score(judgement), -1)


class TestMatchPair(unittest.TestCase):
    def test_get_winner(self):
        judgement = "[[A]]"
        self.assertEqual(
            MatchPair.get_winner(judgement, model_a="model_1", model_b="model_2"),
            "model_1",
        )

        judgement = "[[B]]"
        self.assertEqual(
            MatchPair.get_winner(judgement, model_a="model_1", model_b="model_2"),
            "model_2",
        )

        judgement = "[[A]]"
        self.assertEqual(
            MatchPair.get_winner(judgement, model_a="model_2", model_b="model_1"),
            "model_2",
        )

        judgement = "[[B]]"
        self.assertEqual(
            MatchPair.get_winner(judgement, model_a="model_2", model_b="model_1"),
            "model_1",
        )

        judgement = "[[C]]"
        self.assertEqual(
            MatchPair.get_winner(judgement, model_a="model_1", model_b="model_2"), "tie"
        )

        judgement = "[[D]]"
        self.assertEqual(
            MatchPair.get_winner(judgement, model_a="model_1", model_b="model_2"),
            "error",
        )
