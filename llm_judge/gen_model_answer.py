import argparse
import json
import os
import random
import shortuuid
import time
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


default_temperature_config = {
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
        generation_config (dict): Generation config.
        special_token_map (dict): Special token map. This is used to replace special tokens in the output.
    """
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
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
        for k, v in special_token_map.items():
            output = output.replace(k, v)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument(
        "--benchmark",
        default="jp_bench",
        type=str,
        help="A file that contains instructions (one instruction per line)",
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="random seed for reproducibility"
    )
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    with open(args.config, "r") as f:
        config = json.load(f)

    print("loading model")
    model_name_or_path = config["model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch_dtype
    )
    lora_model_name_or_path = config.get("lora_model_name_or_path")
    if lora_model_name_or_path:
        print("loading peft model")
        model = PeftModel.from_pretrained(model, lora_model_name_or_path)
    model.eval()

    tokenizer_name_or_path = (
        config.get("tokenizer_name_or_path")
        or lora_model_name_or_path
        or model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # test data
    print("loading data")
    data_file = "./data/{}/question.jsonl".format(args.benchmark)
    questions = []
    with open(data_file, "r") as f:
        for line in tqdm(f.read().splitlines()):
            questions.append(json.loads(line))

    print("Start inference.")
    model_id = config["model_id"]
    prompt_template = config["prompt_template"]
    if "{instruction}" not in prompt_template:
        raise ValueError("prompt_template must contain {instruction}")
    special_token_map = config.get("special_token_map", {})
    results = []
    for index, question in tqdm(enumerate(questions)):
        generation_config = config.get("generation_config", {})
        if (
            "temperature" not in generation_config
            or generation_config["temperature"] is None
        ):
            generation_config["temperature"] = default_temperature_config[
                question["category"]
            ]

        instruction = question["turns"][0]
        output = generate_response(
            input_text=prompt_template.format_map({"instruction": instruction}),
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            special_token_map=special_token_map,
        )

        print(f"======={index}=======")
        print(f"Input: {instruction}\n")
        print(f"Output: {output}\n")
        results.append(
            {
                "question_id": int(question["question_id"]),
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": [{"index": 0, "turns": [output]}],
                "tstamp": time.time(),
            }
        )

    predictions_file = "./data/{}/model_answer/{}.jsonl".format(
        args.benchmark, model_id
    )
    dirname = os.path.dirname(predictions_file)
    os.makedirs(dirname, exist_ok=True)
    with open(predictions_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
