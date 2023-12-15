import argparse
import logging
import os

import pandas as pd
import wandb
from common import (
    JUDGEMENT_DIR,
    load_judgements,
    filter_single_judgements,
    filter_pairwise_judgements,
)
from show_result import calculate_win_rate

logger = logging.getLogger(__name__)


def upload_results(
    mode: str,
    result_id: str,
    results: list[dict],
    baseline_model: str = None,
):
    """Upload results to wandb.

    Args:
        mode: Evaluation mode.
        result_id: Result ID.
        results: A list of results.
        baseline_model: Baseline model name. Only used in `pairwise-baseline` mode.
    """
    project = os.getenv("WANDB_PROJECT", "ja-vicuna-qa-benchmark")
    project += f"|{mode}"
    if mode == "pairwise-baseline":
        project += f"|{baseline_model}"
    run = wandb.init(project=project, name=result_id, reinit=True)

    outputs_table = wandb.Table(dataframe=pd.DataFrame(results))
    run.log({"outputs": outputs_table})

    if mode == "pairwise-baseline":
        win_rate_map = calculate_win_rate(results)

        assert baseline_model is not None
        example = results[0]
        if baseline_model == example["model_2"]:
            model = example["model_1"]
            win_rate = win_rate_map["model_1"]["win_rate"]
            adjusted_win_rate = win_rate_map["model_1"]["adjusted_win_rate"]
        else:
            model = example["model_2"]
            win_rate = win_rate_map["model_2"]["win_rate"]
            adjusted_win_rate = win_rate_map["model_2"]["adjusted_win_rate"]

        leaderboard_table = wandb.Table(
            columns=[
                "model",
                "baseline_model",
                "win_rate",
                "adjusted_win_rate",
            ],
            data=[[model, baseline_model, win_rate, adjusted_win_rate]],
        )
        run.log({"leaderboard": leaderboard_table})


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
        "--verbose", "-v", action="count", default=0, help="verbosity level"
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

    logger.info("Login to wandb")
    wandb.login()
    if args.mode != "pairwise-baseline":
        logger.warning(
            "Leaderboard is only available in pairwise-baseline mode. "
            "Only raw outputs will be logged."
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

    logger.info("Log results to wandb")
    for result_id, results in result_id_results_map.items():
        upload_results(
            mode=args.mode,
            result_id=result_id,
            results=results,
            baseline_model=args.baseline_model,
        )
