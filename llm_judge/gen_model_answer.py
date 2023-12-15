import argparse
import json
import logging
import random
import time

import numpy as np
import shortuuid
import torch
from common import PREDICTION_DIR, QUESTION_FILE, load_questions
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE_MAP = {
    "writing": 0.7,
    "roleplay": 0.7,
    "knowledge": 0.001,
    "math": 0.001,
    "coding": 0.001,
    "common-sense": 0.3,
    "counterfactual": 0.7,
    "fermi": 0.3,
    "generic": 0.1,
}


def generate_response(
    input_text, model, tokenizer, generation_config=None, special_token_map=None
):
    """Generate a response from the model given an input text.

    Args:
        input_text (str): Input text.
        model (transformers.PreTrainedModel): Model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer.
        generation_config (Optional[dict]): Generation config.
        special_token_map (Optional[dict]): Special token map used to replace special tokens.
    """
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(model.device)

    input_token_ids = inputs["input_ids"]

    if generation_config is None:
        generation_config = {}

    with torch.no_grad():
        output_token_ids = model.generate(
            **inputs,
            **generation_config,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    output_token_ids = output_token_ids[input_token_ids.size(1) :]

    output = tokenizer.decode(output_token_ids.tolist(), skip_special_tokens=True)
    if special_token_map:
        for src, tgt in special_token_map.items():
            output = output.replace(src, tgt)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument(
        "--bench-name",
        default="jp_bench",
        type=str,
        help="A file that contains instructions (one instruction per line)",
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="random seed for reproducibility"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity level"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite the existing results"
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
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    logger.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)
    logger.debug(config)

    logger.info("Load the model")
    model_name_or_path = config["model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch_dtype
    )
    lora_model_name_or_path = config.get("lora_model_name_or_path")
    if lora_model_name_or_path:
        logger.info("Load the PEFT model")
        model = PeftModel.from_pretrained(model, lora_model_name_or_path)
    model.eval()
    logger.debug(model)

    logger.info("Load the tokenizer")
    tokenizer_name_or_path = (
        config.get("tokenizer_name_or_path")
        or lora_model_name_or_path
        or model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    logger.info("Load the data")
    questions = load_questions(str(QUESTION_FILE))

    logger.info("Start inference.")
    model_id = config["model_id"]
    prompt_template = config["prompt_template"]
    if "{instruction}" not in prompt_template:
        raise ValueError("prompt_template must contain {instruction}")
    special_token_map = config.get("special_token_map", {})

    prediction_dir = PREDICTION_DIR / model_id
    prediction_file = prediction_dir / "results.jsonl"
    if prediction_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"{prediction_file} already exists. Use --overwrite to overwrite."
        )

    results = []
    for index, question in tqdm(enumerate(questions)):
        instruction = question["turns"][0]

        generation_config = config.get("generation_config", {})
        if generation_config.get("temperature") is None:
            category = question["category"]
            generation_config["temperature"] = DEFAULT_TEMPERATURE_MAP[category]

        output = generate_response(
            input_text=prompt_template.format_map({"instruction": instruction}),
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            special_token_map=special_token_map,
        )

        logger.debug(f"======={index}=======")
        logger.debug(f"Input: {instruction}")
        logger.debug(f"Output: {output}")
        results.append(
            {
                "question_id": int(question["question_id"]),
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": [{"index": 0, "turns": [output]}],
                "tstamp": time.time(),
            }
        )

    logger.info("Save the results")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    with open(prediction_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    config_file = prediction_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved the results to {prediction_file}")
