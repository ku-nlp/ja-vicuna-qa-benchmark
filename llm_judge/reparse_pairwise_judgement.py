import argparse
import json
import logging

from common import JUDGEMENT_DIR, MatchPair, load_judgements

logger = logging.getLogger(__name__)


def reparse_result_pairwise(result: dict) -> dict:
    """Reparse the result to determine the winner.

    Args:
        result: A result.
    """
    reparsed_result = result.copy()

    g1_judgment = result["g1_judgment"]
    g1_winner = MatchPair.get_winner(g1_judgment, model_a="model_1", model_b="model_2")
    reparsed_result["g1_winner"] = g1_winner

    g2_judgment = result["g2_judgment"]
    g2_winner = MatchPair.get_winner(g2_judgment, model_a="model_2", model_b="model_1")
    reparsed_result["g2_winner"] = g2_winner

    return reparsed_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    for judgement_dir in (JUDGEMENT_DIR / "pairwise").iterdir():
        result_id_results_map = load_judgements(judgement_dir)
        for result_id, results in result_id_results_map.items():
            reparsed_results = [reparse_result_pairwise(result) for result in results]
            if any(
                result != reparsed_result
                for result, reparsed_result in zip(results, reparsed_results)
            ):
                output_file = judgement_dir / f"{result_id}.jsonl"
                with open(output_file, "w") as f:
                    for result in reparsed_results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                logger.info(f"Fixed {output_file}")
    logger.info("Done")
