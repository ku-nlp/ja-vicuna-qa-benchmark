import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Optional

from common import (
    JUDGEMENT_DIR,
    JUDGEMENT_PROMPT_FILE,
    NEED_REF_CATS,
    PREDICTION_DIR,
    QUESTION_FILE,
    REFERENCE_DIR,
    Judge,
    MatchPair,
    MatchSingle,
    get_model_list,
    load_judge_prompts,
    load_model_answers,
    load_questions,
)
from tqdm import tqdm
from upload_result import upload_results

logger = logging.getLogger(__name__)


def make_match_groups_single(
    questions: list[dict],
    model_answers: dict[str, dict[int, dict]],
    ref_answers: dict[str, dict[int, dict]],
    judge_default: Judge,
    judge_math: Judge,
    num_answers_per_question: Optional[int] = None,
):
    """Make match groups for single answer grading.

    Args:
        questions (list): A list of questions.
        model_answers (dict): A dict of model answers.
        ref_answers (dict): A dict of reference answers.
        judge_default (Judge): A judge for default questions.
        judge_math (Judge): A judge for math questions.
        num_answers_per_question (Optional[int]): Number of answers to evaluate per question.
    """
    match_groups = {}
    for model in model_answers:
        matches = []
        for question in questions:
            qid = question["question_id"]
            answer = model_answers[model][qid]
            if question["category"] in NEED_REF_CATS:
                judge = judge_math
                ref_answer = ref_answers[judge.model][qid]
            else:
                judge = judge_default
                ref_answer = None
            matches.append(
                MatchSingle(
                    question=question,
                    model=model,
                    answer=answer,
                    judge=judge,
                    ref_answer=ref_answer,
                )
            )
        if num_answers_per_question:
            matches = matches[:num_answers_per_question]
        match_groups[f"single:{model}"] = matches
    return match_groups


def make_match_groups_pairwise(
    questions: list[dict],
    model_answers: dict[str, dict[int, dict]],
    ref_answers: dict[str, dict[int, dict]],
    judge_default: Judge,
    judge_math: Judge,
    baseline_model: Optional[str] = None,
    num_answers_per_question: Optional[int] = None,
):
    """Make match groups for pairwise comparison.

    Args:
        questions (list): A list of questions.
        model_answers (dict): A dict of model answers.
        ref_answers (dict): A dict of reference answers.
        judge_default (Judge): A judge for default questions.
        judge_math (Judge): A judge for math questions.
        baseline_model (Optional[str]): The baseline model.
        num_answers_per_question (Optional[int]): Number of answers to evaluate per question.
    """
    match_groups = {}
    for model_1, model_2 in combinations(model_answers, 2):
        if baseline_model and baseline_model not in {model_1, model_2}:
            continue
        matches = []
        for question in questions:
            qid = question["question_id"]
            answer_1 = model_answers[model_1][qid]
            answer_2 = model_answers[model_2][qid]
            if question["category"] in NEED_REF_CATS:
                judge = judge_math
                ref_answer = ref_answers[judge.model][qid]
            else:
                judge = judge_default
                ref_answer = None
            matches.append(
                MatchPair(
                    question=question,
                    model_1=model_1,
                    model_2=model_2,
                    answer_1=answer_1,
                    answer_2=answer_2,
                    judge=judge,
                    ref_answer=ref_answer,
                )
            )
        if num_answers_per_question:
            matches = matches[:num_answers_per_question]
        match_groups[f"pairwise:{model_1}_{model_2}"] = matches
    return match_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="pairwise-baseline",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparison against a baseline. "
            "`pairwise-all` runs pairwise comparison between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4-0613",
        choices=["gpt-4", "gpt-4-0613", "gpt-4-1106-preview", "gpt-3.5-turbo"],
        help="The judge model.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="openai--text-davinci-003",
        help="The baseline model. This is only used in `pairwise-baseline` mode.",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated. If not specified, all models will be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--first-n", type=int, help="Only run the first `n` judgments.")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation and run."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing judgment files."
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Verbosity level"
    )
    parser.add_argument(
        "--num_answers_per_question", type=int, default=None, help="Number of answers to evaluate per question."
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

    if args.wandb:
        import wandb

        wandb.login()
        if args.mode != "pairwise-baseline":
            logger.warning(
                "Leaderboard is only available in pairwise-baseline mode. "
                "Only raw outputs will be logged."
            )

    logger.info("Load questions")
    questions = load_questions(QUESTION_FILE)
    if args.first_n:
        logger.warning(f"Only run the first {args.first_n} judgments")
        questions = questions[: args.first_n]

    logger.info("Load answers")
    if args.model_list is None:
        models = get_model_list(PREDICTION_DIR)
    else:
        models = args.model_list
        if args.mode == "pairwise-baseline" and args.baseline_model not in models:
            models.append(args.baseline_model)
    model_answers = {}
    for model in sorted(models):
        answers = load_model_answers(PREDICTION_DIR / model)
        for question in questions:
            assert question["question_id"] in answers
        model_answers[model] = answers

    logger.info("Load reference answers")
    judge_model = args.judge_model
    answers = load_model_answers(REFERENCE_DIR / "gpt-4")
    for question in filter(lambda x: x["category"] in NEED_REF_CATS, questions):
        assert question["question_id"] in answers
    ref_answers = {judge_model: answers}

    logger.info("Load judge prompts")
    judge_prompts = load_judge_prompts(JUDGEMENT_PROMPT_FILE)

    logger.info("Make matches")
    if args.mode == "single":
        match_groups = make_match_groups_single(
            questions,
            model_answers,
            ref_answers=ref_answers,
            judge_default=Judge(args.judge_model, judge_prompts["single"]),
            judge_math=Judge(args.judge_model, judge_prompts["single-math"]),
            num_answers_per_question=args.num_answers_per_question,
        )
        output_dir = JUDGEMENT_DIR / "single" / args.judge_model
    else:
        assert args.mode in {"pairwise-baseline", "pairwise-all"}
        if args.mode == "pairwise-all":
            baseline_model = None
        else:
            baseline_model = args.baseline_model
        match_groups = make_match_groups_pairwise(
            questions,
            model_answers,
            ref_answers=ref_answers,
            judge_default=Judge(args.judge_model, judge_prompts["pair"]),
            judge_math=Judge(args.judge_model, judge_prompts["pair-math"]),
            baseline_model=baseline_model,
            num_answers_per_question=args.num_answers_per_question,
        )
        output_dir = JUDGEMENT_DIR / "pairwise" / args.judge_model
    target_match_ids = set()
    for match_id in match_groups:
        output_file = output_dir / f"{match_id}.jsonl"
        if output_file.exists():
            if not args.overwrite:
                logger.info(f"Skip {match_id}; to overwrite, use --overwrite")
                continue
        target_match_ids.add(match_id)
    match_groups = {k: v for k, v in match_groups.items() if k in target_match_ids}

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Judge model: {args.judge_model}")
    if args.mode == "pairwise-baseline":
        logger.info(f"Baseline model: {args.baseline_model}")
    logger.info(f"Total number of questions: {len(questions):,}")
    logger.info(
        f"Total number of matches: {sum(len(matches) for matches in match_groups.values()):,}"
    )
    estimated_cost = 0
    for matches in match_groups.values():
        estimated_cost += sum(m.estimate_cost() for m in matches)
    logger.info(f"Total cost (estimated): ${int(estimated_cost):,}")
    logger.info(f"Output directory: {output_dir}")

    if not args.yes:
        input("Press Enter to confirm...")

    logger.info("Play matches")
    for match_id, matches in match_groups.items():
        output_file = output_dir / f"{match_id}.jsonl"
        results = []
        with ThreadPoolExecutor(args.parallel) as executor:
            futures = [executor.submit(match.play) for match in matches]
            for future in tqdm(futures):
                results.append(future.result())

        logger.info(f"Write {len(results)} judgments")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info(f"Saved the judgments to {output_file}")

        if args.wandb:
            logger.info("Log to wandb")
            upload_results(args.mode, match_id, results, args.baseline_model)

