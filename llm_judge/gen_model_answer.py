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
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

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
    "--max_new_tokens", default=None, type=int, help="number of generated tokens"
)
parser.add_argument(
    "--temperature", default=None, type=int, help="generation temperature"
)
parser.add_argument("--tokenizer_path", default=None, type=str)
parser.add_argument(
    "--benchmark",
    default=None,
    type=str,
    help="A file that contains instructions (one instruction per line)",
)
parser.add_argument(
    "--with_prompt",
    action="store_true",
    help="wrap the input with the prompt automatically",
)
parser.add_argument(
    "--interactive",
    action="store_true",
    help="run in the instruction mode (single-turn)",
)
parser.add_argument("--gpus", default="3", type=str)
parser.add_argument(
    "--only_cpu", action="store_true", help="only use CPU for inference"
)
parser.add_argument(
    "--seed", default=0, type=int, help="random seed for reproducibility"
)
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

temperature_config = {
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


# The prompt template below is taken from llama.cpp
# and is slightly different from the one used in training.
# But we find it gives better results
# Japanese version

template = {
    "alpaca": "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:\n\n{instruction}\n\n### Response:\n\n",
    "alpaca_jp": "以下にあるタスクの指示を示します。示された指示に適切に従うように回答を埋めてください。### 指示：\n\n{instruction}\n\n### 回答：\n\n",
    "rinna": "ユーザー: {instruction}<NL>システム: ",
    "llm-jp": "{instruction} ### 回答：",
    "elyza": "[INST] {B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{instruction} [/INST] ",
}


def generate_response(input_text, tokenizer, model, temperature, args):
    if args.temperature is not None:
        temperature = args.temperature
    if "llm-jp" in args.base_model or (args.lora_model and "llm-jp" in args.lora_model):
        if args.with_prompt:
            input_text = "{instruction} ### 回答：".format_map(
                {"instruction": input_text}
            )

        inputs = tokenizer(
            input_text, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                top_p=0.9,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
            )
        s = generation_output[0]
        output = tokenizer.decode(s)
        if args.with_prompt:
            output = output.split("### 回答：")[1].strip()
            output = output.split("<EOD|LLM-jp>")[0].strip()

        return output
    elif "rinna" in args.base_model:
        if args.with_prompt:
            input_text = "ユーザー: {instruction}<NL>システム: ".format_map(
                {"instruction": input_text}
            )

        token_ids = tokenizer.encode(
            input_text, add_special_tokens=False, return_tensors="pt"
        )
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :])
        if args.with_prompt:
            output = output.replace("<NL>", "\n")
            output = output.replace("</s>", "")

        return output
    elif "elyza" in args.base_model:
        if args.with_prompt:
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

        token_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )

        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        output = tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True
        )
        return output

    elif "llama" in args.base_model:
        if args.with_prompt:
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
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=args.max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        if args.with_prompt:
            output = output.split("### Response：")[1].strip()
            output = output.split("\n\n")[0].strip()
        return output


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    tokenizer_path = args.tokenizer_path or args.lora_model or args.base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except ValueError:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, device_map="auto", torch_dtype=torch_dtype
        )
    except ValueError:
        base_model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=False,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model)
    else:
        model = base_model

    # test data
    if args.benchmark is not None:
        question = []
        data_file = "./data/{}/question.jsonl".format(args.benchmark)
        with open(data_file, "r") as f:
            instruction_list = []
            for line in tqdm(f.read().splitlines()):
                tmp_dict = json.loads(line)
                question.append(tmp_dict)
                instruction_list.append(tmp_dict["turns"][0])
            examples = [line.strip() for line in instruction_list]

    # model.eval()

    with torch.no_grad():
        if args.interactive:
            print("Start inference with instruction mode.")

            print("=" * 85)
            print(
                "+ This mode only supports single-turn QA.\n"
                "+ If you want to experience multi-turn dialogue, please use llama.cpp or llamachat."
            )
            print("=" * 85)

            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip()) == 0:
                    break
                output = generate_response(raw_input_text, tokenizer, model, None, args)
                print("Response: ", output)
                print("\n")
        else:
            print("Start inference.")
            results = []
            for index, example in tqdm(enumerate(examples)):
                temperature = temperature_config[question[index]["category"]]
                output = generate_response(example, tokenizer, model, temperature, args)

                response = output
                print(f"======={index}=======")
                print(f"Input: {example}\n")
                print(f"Output: {response}\n")
                results.append(
                    {
                        "question_id": int(question[index]["question_id"]),
                        "answer_id": shortuuid.uuid(),
                        "model_id": args.model_id,
                        "choices": [{"index": 0, "turns": [response]}],
                        "tstamp": time.time(),
                    }
                )
            predictions_file = "./data/{}/model_answer/{}.jsonl".format(
                args.benchmark, args.model_id
            )
            dirname = os.path.dirname(predictions_file)
            os.makedirs(dirname, exist_ok=True)
            with open(predictions_file, "w") as f:
                for tmp_dict in results:
                    json.dump(tmp_dict, f, ensure_ascii=False)
                    f.write("\n")
