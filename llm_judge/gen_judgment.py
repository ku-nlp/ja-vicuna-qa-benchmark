"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import random
from itertools import combinations
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCHMARK_FILE_MAP = {
    "jp_bench": DATA_DIR / "jp_bench" / "question.jsonl",
}
PREDICTION_DIR_MAP = {
    "jp_bench": DATA_DIR / "jp_bench" / "model_answer",
}
REFERENCE_DIR_MAP = {
    "jp_bench": DATA_DIR / "jp_bench" / "reference_answer",
}
JUDGEMENT_DIR_MAP = {
    "jp_bench": DATA_DIR / "jp_bench" / "model_judgment",
}


def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
):
    matches = []
    for question in questions:
        qid = question["question_id"]
        ref_answer = ref_answers[judge.model_name][qid] if ref_answers else None
        for model in models:
            if model == baseline_model:
                continue
            answer = model_answers[model][qid]
            answer_baseline = model_answers[baseline_model][qid]
            matches.append(
                MatchPair(
                    dict(question),
                    model,
                    baseline_model,
                    answer,
                    answer_baseline,
                    judge,
                    ref_answer=ref_answer,
                )
            )
    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
):
    matches = []
    for question in questions:
        qid = question["question_id"]
        ref_answer = ref_answers[judge.model_name][qid] if ref_answers else None
        for model_1, model_2 in combinations(models, 2):
            answer_1 = model_answers[model_1][qid]
            answer_2 = model_answers[model_2][qid]
            matches.append(
                MatchPair(
                    dict(question),
                    model_1,
                    model_2,
                    answer_1,
                    answer_2,
                    judge,
                    ref_answer=ref_answer,
                )
            )
    return matches


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
):
    matches = []
    for question in questions:
        qid = question["question_id"]
        ref_answer = ref_answers[judge.model_name][qid] if ref_answers else None
        for model in models:
            answer = model_answers[model][qid]
            matches.append(
                MatchSingle(
                    dict(question),
                    model,
                    answer,
                    judge,
                    ref_answer=ref_answer,
                )
            )
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    return {
        "default": Judge(judge_model, judge_prompts["pair-v2"]),
        "math": Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True),
    }


def make_judge_single(judge_model, judge_prompts):
    return {
        "default": Judge(judge_model, judge_prompts["single-v1"]),
        "math": Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="jp_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts_jp.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument(
        "--baseline-model", type=str, default="openai--text-davinci-003"
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="random seed for reproducibility"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity level"
    )
    args = parser.parse_args()

    if args.verbose == 0:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(
        format="| %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    logger.info(f"Set random seed to {args.seed}")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    logger.info("Load questions")
    question_file = BENCHMARK_FILE_MAP[args.bench_name]
    questions = load_questions(question_file)
    if args.first_n:
        logger.warning(f"Only run the first {args.first_n} judgments")
        questions = questions[: args.first_n]

    logger.info("Load answers")
    answer_dir = PREDICTION_DIR_MAP[args.bench_name]
    model_answers = load_model_answers(answer_dir)

    logger.info("Load reference answers")
    reference_dir = REFERENCE_DIR_MAP[args.bench_name]
    ref_answers = load_model_answers(reference_dir)

    # Load judge
    logger.info("Load judge prompts")
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    output_dir = JUDGEMENT_DIR_MAP[args.bench_name]

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = output_dir / f"{args.judge_model}_single.jsonl"
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = output_dir / f"{args.judge_model}_pair.jsonl"
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )

    logger.info(f"Benchmark: {args.bench_name}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Baseline model: {baseline_model}")
    logger.info(f"Models: {models}")
    logger.info(f"Total number of questions: {len(questions)}")
    logger.info(f"Total number of matches: {len(matches)}")
    logger.info(f"Output file: {output_file}")
    input("Press Enter to confirm...")

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:
        np.random.shuffle(matches)
        play_a_match_wrapper = partial(play_a_match_func, output_file=output_file)
        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass
