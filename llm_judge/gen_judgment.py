import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import combinations

from common import (
    JUDGEMENT_DIR,
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
    play_a_match_pair,
    play_a_match_single,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
):
    for model in models:
        matches = []
        for question in questions:
            qid = question["question_id"]
            answer = model_answers[model][qid]
            ref_answer = ref_answers[judge.model_name][qid] if ref_answers else None
            matches.append(
                MatchSingle(
                    dict(question),
                    model,
                    answer,
                    judge,
                    ref_answer=ref_answer,
                )
            )
        yield model, matches


def make_match_pairwise(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
):
    for model_1, model_2 in combinations(models, 2):
        if baseline_model and baseline_model not in {model_1, model_2}:
            continue
        matches = []
        for question in questions:
            qid = question["question_id"]
            answer_1 = model_answers[model_1][qid]
            answer_2 = model_answers[model_2][qid]
            answer_ref = ref_answers[judge.model_name][qid] if ref_answers else None
            matches.append(
                MatchPair(
                    dict(question),
                    model_1,
                    model_2,
                    answer_1,
                    answer_2,
                    judge,
                    ref_answer=answer_ref,
                )
            )
        yield f"{model_1}_{model_2}", matches


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
        "--judge-file",
        type=str,
        default="data/judge_prompts_jp.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4",
        choices=["gpt-4", "gpt-3.5-turbo"],
        help="The judge model.",
    )
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
        "--verbose", "-v", action="count", default=0, help="verbosity level"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation and run."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing judgment files."
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

    logger.info("Load questions")
    questions = load_questions(str(QUESTION_FILE))
    if args.first_n:
        logger.warning(f"Only run the first {args.first_n} judgments")
        questions = questions[: args.first_n]
    questions_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    questions_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    logger.info("Load answers")
    model_answers = load_model_answers(str(PREDICTION_DIR))
    for answers in model_answers.values():
        for question in questions:
            assert question["question_id"] in answers

    logger.info("Load reference answers")
    ref_answers = load_model_answers(str(REFERENCE_DIR))
    assert args.judge_model in ref_answers
    for question in filter(lambda x: x["category"] in NEED_REF_CATS, questions):
        assert question["question_id"] in ref_answers[args.judge_model]

    logger.info("Load judge prompts")
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.model_list is None:
        models = get_model_list(str(PREDICTION_DIR))
    else:
        models = args.model_list

    logger.info("Make matches")
    match_groups = {}
    if args.mode == "single":
        judge_default = Judge(args.judge_model, judge_prompts["single"])
        judge_math = Judge(
            args.judge_model, judge_prompts["single-math"], ref_based=True
        )
        play_a_match_func = play_a_match_single
        output_dir = JUDGEMENT_DIR / "single" / args.judge_model
        make_match_func = make_match_single
        baseline_model = None
    else:
        assert args.mode in {"pairwise-baseline", "pairwise-all"}
        judge_default = Judge(args.judge_model, judge_prompts["pair"])
        judge_math = Judge(args.judge_model, judge_prompts["pair-math"], ref_based=True)
        play_a_match_func = play_a_match_pair
        output_dir = JUDGEMENT_DIR / "pairwise" / args.judge_model
        make_match_func = make_match_pairwise
        if args.mode == "pairwise-all":
            baseline_model = None
        else:
            baseline_model = args.baseline_model
    for match_id, matches in make_match_func(
        questions_default, models, model_answers, judge_default, baseline_model
    ):
        match_groups[match_id] = matches
    for match_id, matches in make_match_func(
        questions_math, models, model_answers, judge_math, baseline_model
    ):
        match_groups[match_id] += matches
    target_match_ids = set()
    for match_id in match_groups:
        output_file = output_dir / f"{match_id}.jsonl"
        if output_file.exists():
            if args.overwrite:
                output_file.unlink()
            else:
                logger.info(
                    f"Skip evaluating {match_id}; to overwrite, use --overwrite"
                )
                continue
        target_match_ids.add(match_id)
    match_groups = {k: v for k, v in match_groups.items() if k in target_match_ids}

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Baseline model: {baseline_model}")
    logger.info(f"Models: {models}")
    logger.info(f"Total number of questions: {len(questions):,}")
    logger.info(
        f"Total number of matches: {sum(len(m) for m in match_groups.values()):,}"
    )
    logger.info(f"Output file: {output_dir}")

    if not args.yes:
        input("Press Enter to confirm...")

    logger.info("Play matches")
    for match_id, matches in match_groups.items():
        output_file = output_dir / f"{match_id}.jsonl"
        if args.parallel == 1:
            for match in tqdm(matches):
                play_a_match_func(match, output_file=str(output_file))
        else:
            play_a_match_wrapper = partial(
                play_a_match_func, output_file=str(output_file)
            )
            with ThreadPoolExecutor(args.parallel) as executor:
                for match in tqdm(
                    executor.map(play_a_match_wrapper, matches), total=len(matches)
                ):
                    pass
