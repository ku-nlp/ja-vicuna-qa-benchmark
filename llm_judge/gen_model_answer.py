import argparse
import json
import os
import random
import shortuuid
import time
from tqdm import tqdm

import numpy as np
import torch
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
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


def generate_response(input_text, tokenizer, model, temperature, max_new_tokens, args):
    if "llm-jp" in args.base_model or (args.lora_model and "llm-jp" in args.lora_model):
        input_text = "{instruction} ### 回答：".format_map({"instruction": input_text})

        inputs = tokenizer(
            input_text, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        input_token_ids = inputs["input_ids"]

        output_token_ids = model.generate(
            **inputs,
            top_p=0.9,
            temperature=temperature,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
        output_token_ids = output_token_ids[input_token_ids.size(1) :]
        output = tokenizer.decode(output_token_ids.tolist(), skip_special_tokens=True)
        return output
    elif "rinna" in args.base_model:
        input_text = "ユーザー: {instruction}<NL>システム: ".format_map(
            {"instruction": input_text}
        )

        input_token_ids = tokenizer.encode(
            input_text, add_special_tokens=False, return_tensors="pt"
        )
        output_token_ids = model.generate(
            input_token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
        output_token_ids = output_token_ids[input_token_ids.size(1) :]
        output = tokenizer.decode(output_token_ids.tolist(), skip_special_tokens=True)
        output = output.replace("<NL>", "\n")
        return output
    elif "elyza" in args.base_model:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

        prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
            bos_token=tokenizer.bos_token,
            b_inst=B_INST,
            system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
            prompt=input_text,
            e_inst=E_INST,
        )

        input_token_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )

        output_token_ids = model.generate(
            input_token_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        output = tokenizer.decode(
            output_token_ids.tolist()[0][input_token_ids.size(1) :],
            skip_special_tokens=True,
        )
        return output

    elif "llama" in args.base_model:
        input_text = "以下にあるタスクの指示を示します。示された指示に適切に従うように回答を埋めてください。### 指示：\n\n{instruction}\n\n### 回答：\n\n".format_map(
            {"instruction": input_text}
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            no_repeat_ngram_size=3,
        )

        with torch.no_grad():
            output_token_ids = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = output_token_ids.sequences[0]
        output = tokenizer.decode(s)
        output = output.split("### Response：")[1].strip()
        output = output.split("\n\n")[0].strip()
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=None, type=str, required=True)
    parser.add_argument(
        "--lora_model",
        default=None,
        type=str,
        help="If None, perform inference on the base model",
    )
    parser.add_argument("--model_id", default=None, type=str, help="name of the model")
    parser.add_argument(
        "--max_new_tokens", default=512, type=int, help="number of generated tokens"
    )
    parser.add_argument(
        "--temperature", default=None, type=int, help="generation temperature"
    )
    parser.add_argument("--tokenizer_path", default=None, type=str)
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    tokenizer_path = args.tokenizer_path or args.lora_model or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", torch_dtype=torch_dtype
    )
    if args.lora_model:
        print("loading peft model")
        model = PeftModel.from_pretrained(model, args.lora_model)
    model.eval()

    # test data
    print("loading data")
    data_file = "./data/{}/question.jsonl".format(args.benchmark)
    questions = []
    with open(data_file, "r") as f:
        for line in tqdm(f.read().splitlines()):
            questions.append(json.loads(line))

    print("Start inference.")
    results = []
    for index, question in tqdm(enumerate(questions)):
        instruction = question["turns"][0]
        temperature = (
            args.temperature or default_temperature_config[question["category"]]
        )
        max_new_tokens = args.max_new_tokens
        with torch.no_grad():
            output = generate_response(
                instruction, tokenizer, model, temperature, max_new_tokens, args
            )

        print(f"======={index}=======")
        print(f"Input: {instruction}\n")
        print(f"Output: {output}\n")
        results.append(
            {
                "question_id": int(question["question_id"]),
                "answer_id": shortuuid.uuid(),
                "model_id": args.model_id,
                "choices": [{"index": 0, "turns": [output]}],
                "tstamp": time.time(),
            }
        )

    predictions_file = "./data/{}/model_answer/{}.jsonl".format(
        args.benchmark, args.model_id
    )
    dirname = os.path.dirname(predictions_file)
    os.makedirs(dirname, exist_ok=True)
    with open(predictions_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
