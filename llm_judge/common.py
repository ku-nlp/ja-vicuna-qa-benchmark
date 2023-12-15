"""
Common data structures and utilities.
"""
import ast
import dataclasses
import json
import logging
import os
import re
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

# Data paths
JP_BENCH_DIR = Path(__file__).resolve().parent.parent / "data" / "jp_bench"
QUESTION_FILE = JP_BENCH_DIR / "question.jsonl"
PREDICTION_DIR = JP_BENCH_DIR / "model_answer"
REFERENCE_DIR = JP_BENCH_DIR / "reference_answer"
JUDGEMENT_DIR = JP_BENCH_DIR / "model_judgment"

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

TIE_DELTA = 0.1

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding"]

# Extract scores from judgments
two_score_pattern = re.compile(r"\[\[(\d+\.?\d*),\s?(\d+\.?\d*)]]")
two_score_pattern_backup = re.compile(r"\[(\d+\.?\d*),\s?(\d+\.?\d*)]")
one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)]]")
one_score_pattern_another_format = re.compile(r"\[\[rating:(\d+)]]")
one_score_pattern_another_format2 = re.compile(r"\[\[rating: (\d+)]]")

reverse_model_map = {"model_1": "model_2", "model_2": "model_1"}


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict

    def judge(self, **kwargs):
        messages = [
            {"role": "system", "content": self.prompt_template["system_prompt"]},
            {
                "role": "user",
                "content": self.prompt_template["prompt_template"].format(**kwargs),
            },
        ]
        for _ in range(API_MAX_RETRY):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=2048,
                )
                return response["choices"][0]["message"]["content"]
            except openai.error.OpenAIError as e:
                logger.warning(e)
                time.sleep(API_RETRY_SLEEP)


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: dict = None

    def __post_init__(self) -> None:
        if self.judge.prompt_template["type"] != "single":
            raise ValueError(
                f"invalid judge type: {self.judge.prompt_template['type']}"
            )

        if self.judge.prompt_template["output_format"] != "[[rating]]":
            raise ValueError(
                f"Invalid output format: {self.judge.prompt_template['output_format']}"
            )

    def play(self):
        kwargs = {
            "question": self.question["turns"][0],
            "answer": self.answer["choices"][0]["turns"][0],
        }
        if self.ref_answer is not None:
            kwargs["ref_answer_1"] = self.ref_answer["choices"][0]["turns"][0]
        judgment = self.judge.judge(**kwargs)
        match = (
            re.search(one_score_pattern, judgment)
            or re.search(one_score_pattern_another_format, judgment)
            or re.search(one_score_pattern_another_format2, judgment)
        )
        if match:
            score = ast.literal_eval(match.groups()[0])
        else:
            score = -1

        return {
            "question_id": self.question["question_id"],
            "model": self.model,
            "judge": (self.judge.model_name, self.judge.prompt_template["name"]),
            "judgment": judgment,
            "score": score,
            "turn": 1,
            "tstamp": time.time(),
        }


@dataclasses.dataclass
class MatchPair:
    question: dict
    model_1: str
    model_2: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: dict = None

    def __post_init__(self) -> None:
        if self.judge.prompt_template["type"] != "pairwise":
            raise ValueError(
                f"invalid judge type: {self.judge.prompt_template['type']}"
            )

        if self.judge.prompt_template["output_format"] != "[[A]]":
            raise ValueError(
                f"Invalid output format: {self.judge.prompt_template['output_format']}"
            )

    def play(self):
        def play(answer_a, answer_b):
            kwargs = {
                "question": self.question["turns"][0],
                "answer_a": answer_a["choices"][0]["turns"][0],
                "answer_b": answer_b["choices"][0]["turns"][0],
            }
            if self.ref_answer is not None:
                kwargs["ref_answer_1"] = self.ref_answer["choices"][0]["turns"][0]
            judgment = self.judge.judge(**kwargs)

            if "[[A]]" in judgment:
                winner = "A"
            elif "[[B]]" in judgment:
                winner = "B"
            elif "[[C]]" in judgment:
                winner = "tie"
            else:
                winner = "error"

            return winner, judgment

        g1_winner, g1_judgment = play(self.answer_1, self.answer_2)
        g2_winner, g2_judgment = play(self.answer_2, self.answer_1)

        g1_map = {"A": "model_1", "B": "model_2"}
        g2_map = {"A": "model_2", "B": "model_1"}
        g1_winner = g1_map.get(g1_winner, g1_winner)
        g2_winner = g2_map.get(g2_winner, g2_winner)

        result = {
            "question_id": self.question["question_id"],
            "model_1": self.model_1,
            "model_2": self.model_2,
            "g1_winner": g1_winner,
            "g2_winner": g2_winner,
            "judge": (self.judge.model_name, self.judge.prompt_template["name"]),
            "g1_judgment": g1_judgment,
            "g2_judgment": g2_judgment,
            "turn": 1,
            "tstamp": time.time(),
        }
        return result


def load_questions(question_file: str) -> list[dict]:
    """Load questions from a file."""
    with open(question_file, "r") as fin:
        return [json.loads(line) for line in fin]


def load_model_answers(answer_dir: str):
    """Load model answers."""
    answers = {}
    with open(Path(answer_dir) / "results.jsonl", "r") as fin:
        for line in fin:
            line = json.loads(line)
            answers[line["question_id"]] = line
    return answers


def load_judge_prompts(prompt_file: str):
    """Load judge prompts."""
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def get_model_list(answer_dir):
    """Get model list from answer directory."""
    return [path.name for path in Path(answer_dir).iterdir()]
