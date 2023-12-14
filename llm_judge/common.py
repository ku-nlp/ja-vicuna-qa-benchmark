"""
Common data structures and utilities.
"""
import ast
import dataclasses
import glob
import json
import logging
import os
import re
import time

import openai
from dotenv import load_dotenv
from model_adapter import get_conversation_template

logger = logging.getLogger(__name__)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

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


def chat_completion_openai(model, conv, temperature, max_tokens):
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response["choices"][0]["message"]["content"]
        except openai.error.OpenAIError as e:
            logger.warning(e)
            time.sleep(API_RETRY_SLEEP)


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False


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
        system_prompt = self.judge.prompt_template["system_prompt"]

        kwargs = {}
        if self.ref_answer is not None:
            kwargs["ref_answer_1"] = self.ref_answer["choices"][0]["turns"][0]
        user_prompt = self.judge.prompt_template["prompt_template"].format(
            question=self.question["turns"][0],
            answer=self.answer["choices"][0]["turns"][0],
            **kwargs,
        )

        conv = get_conversation_template(self.judge.model_name)
        conv.system = system_prompt
        conv.append_message(conv.roles[0], user_prompt)

        judgment = chat_completion_openai(
            self.judge.model_name, conv, temperature=0, max_tokens=2048
        )
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
            "user_prompt": user_prompt,
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
            system_prompt = self.judge.prompt_template["system_prompt"]

            kwargs = {}
            if self.ref_answer is not None:
                kwargs["ref_answer_1"] = self.ref_answer["choices"][0]["turns"][0]
            user_prompt = self.judge.prompt_template["prompt_template"].format(
                question=self.question["turns"][0],
                answer_a=answer_a["choices"][0]["turns"][0],
                answer_b=answer_b["choices"][0]["turns"][0],
                **kwargs,
            )

            conv = get_conversation_template(self.judge.model_name)
            conv.system = system_prompt
            conv.append_message(conv.roles[0], user_prompt)

            judgment = chat_completion_openai(
                self.judge.model_name, conv, temperature=0, max_tokens=2048
            )

            if "[[A]]" in judgment:
                winner = "A"
            elif "[[B]]" in judgment:
                winner = "B"
            elif "[[C]]" in judgment:
                winner = "tie"
            else:
                winner = "error"

            return winner, user_prompt, judgment

        g1_winner, g1_user_prompt, g1_judgment = play(self.answer_1, self.answer_2)
        g2_winner, g2_user_prompt, g2_judgment = play(self.answer_2, self.answer_1)

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
            "g1_user_prompt": g1_user_prompt,
            "g1_judgment": g1_judgment,
            "g2_user_prompt": g2_user_prompt,
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
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    model_answers = {}
    for filename in sorted(filenames):
        logger.debug(f"Loading model answers from {filename}")
        model_name, _ = os.path.splitext(os.path.basename(filename))
        answer = {}
        with open(filename, "r") as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer
    return model_answers


def load_judge_prompts(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def play_a_match_single(match: MatchSingle, output_file: str):
    result = match.play()
    logger.debug(
        f"Question: {result['question_id']}, "
        f"Model: {result['model']}, "
        f"Score: {result['score']}"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def play_a_match_pair(match: MatchPair, output_file: str):
    result = match.play()
    logger.debug(
        f"Question: {result['question_id']}, "
        f"Model 1: {result['model_1']}, "
        f"Model 2: {result['model_2']}, "
        f"Winner 1: {result['g1_winner']}, "
        f"Winner 2: {result['g2_winner']}"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a") as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names
