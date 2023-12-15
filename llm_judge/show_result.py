import argparse
import logging
from pathlib import Path

import pandas as pd
from common import (
    JUDGEMENT_DIR,
    load_judgements,
    filter_single_judgements,
    filter_pairwise_judgements,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JUDGEMENT_DIR_MAP = {
    "jp_bench": DATA_DIR / "jp_bench" / "model_judgment",
}

pd.set_option("display.max_colwidth", 1000)


def calculate_average_score(results: list[dict]):
    """Calculate average score.

    Args:
        results: A list of results.
    """
    score = sum([result["score"] for result in results]) / len(results)
    return score


def calculate_win_rate(results: list[dict]):
    """Calculate win rate and adjusted win rate.

    Args:
        results: A list of results.
    """
    num_win = 0
    num_tie = 0
    for result in results:
        if result["g1_winner"] == "tie" or result["g1_winner"] != result["g2_winner"]:
            num_tie += 1
        elif result["g1_winner"] == "model_1":
            num_win += 1
    win_rate = num_win / len(results)
    adjusted_win_rate = (num_win + 0.5 * num_tie) / len(results)
    return {
        "model_1": {"win_rate": win_rate, "adjusted_win_rate": adjusted_win_rate},
        "model_2": {
            "win_rate": 1 - win_rate,
            "adjusted_win_rate": 1 - adjusted_win_rate,
        },
    }


def display_result_single(result_id_results_map: dict[str, list[dict]]):
    """Display single answer grading results.

    Args:
        result_id_results_map: A map from result id to a list of results.
    """
    result_table = []
    for _, results in result_id_results_map.items():
        example = results[0]
        score = calculate_average_score(results)
        result_table.append(
            {
                "model": example["model"],
                "score": score,
            }
        )
    df = pd.DataFrame(result_table)
    print(df.sort_values(by="score", ascending=False))


def display_result_pairwise(
    result_id_results_map: dict[str, list[dict]], baseline_model=None
):
    """Display pairwise win rate results.

    Args:
        result_id_results_map: A map from result id to a list of results.
        baseline_model: Baseline model. If not specified, all pairs will be compared.
    """
    result_table = []
    for _, results in result_id_results_map.items():
        example = results[0]
        win_rate_map = calculate_win_rate(results)
        if baseline_model:
            if baseline_model == example["model_2"]:
                model_1 = example["model_1"]
                model_2 = example["model_2"]
                win_rate = win_rate_map["model_1"]["win_rate"]
                adjusted_win_rate = win_rate_map["model_1"]["adjusted_win_rate"]
            else:
                model_1 = example["model_2"]
                model_2 = example["model_1"]
                win_rate = win_rate_map["model_2"]["win_rate"]
                adjusted_win_rate = win_rate_map["model_2"]["adjusted_win_rate"]
        else:
            model_1 = example["model_1"]
            model_2 = example["model_2"]
            win_rate = win_rate_map["model_1"]["win_rate"]
            adjusted_win_rate = win_rate_map["model_1"]["adjusted_win_rate"]
        result_table.append(
            {
                "model_1": model_1,
                "model_2": model_2,
                "win_rate": win_rate,
                "adjusted_win_rate": adjusted_win_rate,
            }
        )

    df = pd.DataFrame(result_table)
    print(df.sort_values(by="adjusted_win_rate", ascending=False))


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
    parser.add_argument("--judge-model", type=str, default="gpt-4", help="Judge model")
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="openai--text-davinci-003",
        help="Baseline model. Only used in `pairwise-baseline` mode.",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Verbosity level"
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

    logger.info("Load judgements")
    mode = "single" if args.mode == "single" else "pairwise"
    judgement_dir = JUDGEMENT_DIR / mode / args.judge_model
    result_id_results_map = load_judgements(judgement_dir)
    if args.mode == "single":
        result_id_results_map = filter_single_judgements(
            result_id_results_map, args.model_list
        )
    elif args.mode == "pairwise-baseline":
        result_id_results_map = filter_pairwise_judgements(
            result_id_results_map, args.model_list, args.baseline_model
        )
    else:
        result_id_results_map = filter_pairwise_judgements(
            result_id_results_map, args.model_list
        )

    if args.mode == "single":
        display_result_single(result_id_results_map)
    elif args.mode == "pairwise-baseline":
        display_result_pairwise(result_id_results_map, args.baseline_model)
    elif args.mode == "pairwise-all":
        display_result_pairwise(result_id_results_map)
