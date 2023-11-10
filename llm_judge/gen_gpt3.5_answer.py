from typing import List
import openai
import json
import os
from tqdm import tqdm
import sys
import time
import shortuuid
#use following codes to activate azure:
#openai.api_type = "azure"
#openai.api_base = "https://pomegranate.openai.azure.com/"
#openai.api_version = "2022-12-01"
openai.api_key = "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"

#use openai api with group PRISM
class Chat_Demo(object):
    def __init__(self, model, user_system, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, n, stream, stop, logit_bias):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.n = n
        self.stream = stream
        self.stop = stop
        self.logit_bias = logit_bias
        self.chat_list = []
        self.chat_list = [{"role": "system", "content": user_system}]

    def get_chat_output(self, user_prompt):
        self.chat_list.append({"role": "user", "content": user_prompt})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.chat_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            n=self.n,
            stream=self.stream,
            stop=self.stop
        )
        reply = response['choices'][0]['message']['content']
        self.chat_list.append(response['choices'][0]['message'])

        return reply

    def delete_last_chat(self):
        self.chat_list.pop()
        self.chat_list.pop()

# usage template for chatgpt
def run_chatgpt(user_prompt_list):
    demo = Chat_Demo(
        model="gpt-3.5-turbo-16k-0613",  # gpt-3.5-turbo: chatgpt with lowerest price, gpt-4: lateset version, higher price
        #model="gpt-4",  # gpt-3.5-turbo: chatgpt with lowerest price, gpt-4: lateset version, higher price
        user_system="You are a helpful assistant", # add more description after this to fit your task, e.g., "you are a helpful assistant that translates English to Chinese." will be a good system for MT.
        temperature=1.0,  # default 1.0, control randomness: lowring results in less random completion (0 ~ 2.0)
        max_tokens=256,  # max number of tokens to generate (1 ~ 4,000)
        top_p=1,  # default 1, control diversity (0 ~ 1.0), openai suggests not to alter this with temperature together
        frequency_penalty=0,  # default 0, how to penalize new tokens based on their existing frequency (-2.0 ~ 2.0)
        presence_penalty=0,  # default 0, 这个是对于词是否已经出现过的惩罚，文档上说这个值调高可以增大谈论新topic的概率 (-2.0 ~ 2.0)
        stop="", # manually control where to stop
        stream=False, # temporarily keep false
        n=1, # n choices of reply for each input
        logit_bias=None # use the bias to control what tokens you want to appear or disappear, this function is NOT implemented in Chat_Demo now
    )
    reply = demo.get_chat_output(user_prompt_list)
    return reply
if __name__ == '__main__':

    question = []
    data_file = "./data/question_Japanese_ver.jsonl"
    with open(data_file,'r') as f:
        instruction_list = []
        for line in tqdm(f.read().splitlines()):
            tmp_dict = json.loads(line)
            question.append(tmp_dict)
            instruction_list.append(tmp_dict["text"][0:])
        #examples = [l.strip() for l in instruction_list]
        examples= instruction_list
    results = []
    for index, example in tqdm(enumerate(examples)):
        response=run_chatgpt(example)
        results.append({"question_id":question[index]["question_id"],"answer_id":shortuuid.uuid(),"model_id":"gpt-4" ,"choices":[{"index": 0, "turns": [response]}],"tstamp": time.time(),})    
    predictions_file = "./data/jp_bench/model_answer/GPT-4_reference.jsonl"
    dirname = os.path.dirname(predictions_file)
    os.makedirs(dirname,exist_ok=True)
    with open(predictions_file, 'w') as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')