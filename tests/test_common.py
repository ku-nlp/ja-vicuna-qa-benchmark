import unittest

from llm_judge.common import MatchSingle


class TestMatchSingle(unittest.TestCase):
    def test_get_score(self) -> None:
        judgement = "[[1]]"
        self.assertEqual(MatchSingle.get_score(judgement), 1)

        judgement = "[[2]]"
        self.assertEqual(MatchSingle.get_score(judgement), 2)

        judgement = "[[rating:3]]"
        self.assertEqual(MatchSingle.get_score(judgement), 3)

        judgement = "[[rating: 4.5]]"
        self.assertEqual(MatchSingle.get_score(judgement), -1)

        judgement = "[[rating: Perfect]]"
        self.assertEqual(MatchSingle.get_score(judgement), -1)
