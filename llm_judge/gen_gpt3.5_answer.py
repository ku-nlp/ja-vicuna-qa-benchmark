import openai
import json
import os
from tqdm import tqdm
import time
import shortuuid

from typing import List

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


class GPT3_Demo(object):
    def __init__(
        self,
        engine,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        best_of,
        logprobs,
    ):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of
        self.logprobs = logprobs

    def get_multiple_sample(self, prompt_list: List[str]):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            best_of=self.best_of,
            logprobs=self.logprobs,
        )
        results = [choice.text for choice in response.choices]
        return results


def run_gpt3(prompt_list):
    demo = GPT3_Demo(
        engine="text-davinci-003",  # text-davinci-003: best, text-ada-001: lowest price
        temperature=0,  # control randomness: lowring results in less random completion (0 ~ 1.0)
        max_tokens=2048,  # max number of tokens to generate (1 ~ 4,000)
        top_p=1,  # control diversity (0 ~ 1.0)
        frequency_penalty=0,  # how to penalize new tokens based on their existing frequency (0 ~ 2.0)
        presence_penalty=0,  # 这个是对于词是否已经出现过的惩罚，文档上说这个值调高可以增大谈论新topic的概率 (0 ~ 2.0)
        best_of=1,  # 这个是说从多少个里选最好的，如果这里是10，就会生成10个然后选最好的，但是这样会更贵(1 ~ 20)
        logprobs=1,
    )
    results = demo.get_multiple_sample(prompt_list)
    return results


if __name__ == "__main__":
    data_file = "./data/jp_bench/question.jsonl"
    with open(data_file) as f:
        questions = [json.loads(line) for line in tqdm(f)]

    results = []
    for question in tqdm(questions):
        instruction = question["turns"][0]
        response = run_gpt3(instruction)
        results.append(
            {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": "gpt-3.5-davinci",
                "choices": [{"index": 0, "turns": response}],
                "tstamp": time.time(),
            }
        )

    predictions_file = "./data/jp_bench/model_answer/openai--text-davinci-003.jsonl"
    dirname = os.path.dirname(predictions_file)
    os.makedirs(dirname, exist_ok=True)
    with open(predictions_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
