import argparse
import json
import logging
import os
import time

import openai
import shortuuid
from common import PREDICTION_DIR, QUESTION_FILE
from dotenv import load_dotenv
from tqdm import tqdm

logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


def generate_response(input_text, generation_config) -> str:
    response = openai.Completion.create(prompt=input_text, **generation_config)
    return response.choices[0].text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
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

    logger.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)
    logger.debug(config)

    logger.info("Load the data")
    with open(QUESTION_FILE) as f:
        questions = [json.loads(line) for line in tqdm(f)]

    logger.info("Start inference.")
    model_id = config["model_id"]
    if "generation_config" not in config:
        raise ValueError("'generation_config' is not found in the config file.")
    prompt_template = config["prompt_template"]
    if "{instruction}" not in prompt_template:
        raise ValueError("prompt_template must contain {instruction}")
    generation_config = config["generation_config"]

    results = []
    for question in tqdm(questions):
        instruction = question["turns"][0]
        output = generate_response(
            input_text=prompt_template.format_map({"instruction": instruction}),
            generation_config=generation_config,
        )

        logger.debug(f"{instruction}\n\n{output}")

        results.append(
            {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": [{"index": 0, "turns": output}],
                "tstamp": time.time(),
            }
        )

    logger.info("Save the results")
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    prediction_file = PREDICTION_DIR / f"{model_id}.json"
    with open(prediction_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Saved the results to {prediction_file}")
