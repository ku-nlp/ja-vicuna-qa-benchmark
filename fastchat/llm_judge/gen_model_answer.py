import argparse
import json, os
import shortuuid
import time
from tqdm import tqdm
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--model_id', default=None, type=str,help="name of the model")
parser.add_argument('--max_new_tokens', default=None, type=int,help="number of generated tokens")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--benchmark',default=None, type=str,help="A file that contains instructions (one instruction per line)")
parser.add_argument('--with_prompt',action='store_true',help="wrap the input with the prompt automatically")
parser.add_argument('--interactive',action='store_true',help="run in the instruction mode (single-turn)")
parser.add_argument('--gpus', default=" 3", type=str)
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import GenerationConfig,LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import  PeftModel


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}


 # The prompt template below is taken from llama.cpp
 # and is slightly different from the one used in training.
 # But we find it gives better results
 #Japanese version 
prompt_input = (
    "以下にあるタスクの指示を示します。"
    "示された指示に適切に従うように回答を埋めてください。"
    "### 指示：\n\n{instruction}\n\n### 回答：\n\n"
)
prompt_input_alpaca = ("Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:\n\n{instruction}\n\n### Response:\n\n")

prompt_input_jp = (
    "ユーザー: {instruction}<NL>システム: "
)

sample_data = ["なぜ公害を減らし、環境を守ることが重要なのか？"]

test_prompt = [
    {
        "speaker": "ユーザー",
        "text": "コンタクトレンズを慣れるにはどうすればよいですか？"
    },
    {
        "speaker": "システム",
        "text": "これについて具体的に説明していただけますか？何が難しいのでしょうか？"
    },
    {
        "speaker": "ユーザー",
        "text": "目が痛いのです。"
    },
    {
        "speaker": "システム",
        "text": "分かりました、コンタクトレンズをつけると目がかゆくなるということですね。思った以上にレンズを外す必要があるでしょうか？"
    },
    {
        "speaker": "ユーザー",
        "text": "いえ、レンズは外しませんが、目が赤くなるんです。"
    }
]

def rinna_prompt(tmp_dict):
    prompt = [
        f"{uttr['speaker']}: {uttr['text']}"
        for uttr in prompt
    ]
    prompt = "<NL>".join(prompt)
    prompt = (
        prompt
        + "<NL>"
        + "システム: "
    )
def generate_prompt(instruction, base_model, input=None):
    if input:

        instruction = instruction + '\n' + input
    if "rinna" in base_model:
        return prompt_input_jp.format_map({'instruction': instruction})
    elif "llama" in base_model.lowercase():
        return prompt_input_alpaca.format_map({'instruction': instruction})
    else:
        return prompt_input.format_map({'instruction': instruction})


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )
    except:
        base_model = LlamaForCausalLM.from_pretrained(
        args.base_model, 
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',)
    else:
        model = base_model

    if device==torch.device('cpu'):
        model.float()
    # test data
    #args.benchmark= 'question'
    if args.benchmark is None:
        examples = sample_data
    else:
        question = []
        data_file = "./data/{}/question.jsonl".format(args.benchmark)
        with open(data_file,'r') as f:
            instruction_list = []
            for line in tqdm(f.read().splitlines()):
                tmp_dict = json.loads(line)
                question.append(tmp_dict)
                instruction_list.append(tmp_dict["turns"][0])
            examples = [l.strip() for l in instruction_list]
            #examples= instruction_list
            
            
        '''print("first 10 examples:")
        for example in examples[:10]:
            print(example)'''
    #model.eval()

    with torch.no_grad():
        if args.interactive:
            print("Start inference with instruction mode.")

            print('='*85)
            print("+ 该模式下仅支持单轮问答，无多轮对话能力。\n"
                  "+ 如要进行多轮对话，请使用llama.cpp或llamachat工具。")
            print('-'*85)
            print("+ This mode only supports single-turn QA.\n"
                  "+ If you want to experience multi-turn dialogue, please use llama.cpp or llamachat.")
            print('='*85)

            while True:
                raw_input_text = input("Input:")
                #print(raw_input_text)
                if len(raw_input_text.strip())==0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text,base_model=args.base_model)
                else:
                    input_text = raw_input_text
                if False:
                    print (input_text)   
                    inputs = tokenizer(input_text, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)
                    generation_config = GenerationConfig(
                        temperature=0.1,
                        top_p=0.75,
                        top_k=40,
                        num_beams=4,
                        no_repeat_ngram_size=3
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
                    print(output)
                else:
                    print (input_text)   
                    inputs = tokenizer(input_text, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)

                    with torch.no_grad():
                        generation_output = model.generate(
                            input_ids=input_ids,
                            top_p=1.0,
                            top_k=0,
                            temperature=0.1,
                            repetition_penalty=1.0,
                            return_dict_in_generate=True,
                            output_scores=True,
                            max_new_tokens=args.max_new_tokens,
                            )
                    s = generation_output.sequences[0]
                    output = tokenizer.decode(s)
                    output = output.split("### 回答：")[1].strip()
                    output = output.split("\n\n")[0].strip()
                    print(output)
                    #output = output.split("### response:")[1].strip()
                    #output = output.split("\n\n")[0].strip()
                #print(input_text)
                # 对input编码 get answer
                '''token_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
                output_ids = model.generate(
                    token_ids.to(model.device),
                    do_sample=True,
                    max_new_tokens=128,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                output = tokenizer.decode(generation_output.tolist()[0][token_ids.size(1):])
                output = output.replace("<NL>", "\n")
                response = output
                '''
                #print("Response: ",output)
                #print("\n")
        else:
            print("Start inference.")
            results = []
            for index, example in tqdm(enumerate(examples)):
                temperature = temperature_config[question[index]["category"]]
                if args.with_prompt is True:
                    input_text = generate_prompt(instruction=example,base_model=args.base_model)
                else:
                    input_text = example
                if "rinna" in args.base_model:
                    token_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
                    output_ids = model.generate(
                    token_ids.to(model.device),
                    do_sample=True,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                    )
                    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
                    output = output.replace("<NL>", "\n")
                    output = output.replace("</s>", "")
                elif "llama" in args.base_model.lowercase():
                    inputs = tokenizer(input_text, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=0.75,
                        top_k=40,
                        num_beams=4,
                        no_repeat_ngram_size=3
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
                    output = output.split("### Response：")[1].strip()
                    output = output.split("\n\n")[0].strip()

                else:
                    inputs = tokenizer(input_text, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)

                    with torch.no_grad():
                        generation_output = model.generate(
                            input_ids=input_ids,
                            top_p=1.0,
                            top_k=0,
                            temperature=temperature,
                            repetition_penalty=1.0,
                            return_dict_in_generate=True,
                            output_scores=True,
                            max_new_tokens=args.max_new_tokens,
                            )
                    s = generation_output.sequences[0]
                    output = tokenizer.decode(s)
                    output = output.split("### 回答：")[1].strip()
                    output = output.split("\n\n")[0].strip()

                response = output
                #print("Response: ",response)
                #print("\n")

                #s = generation_output[0]
                #output = tokenizer.decode(s,skip_special_tokens=True)
                '''if args.with_prompt:
                    response = output.split("### Response:")[1].strip()'''
                
                # response = output
                print(f"======={index}=======")
                print(f"Input: {example}\n")
                print(f"Output: {response}\n")
                results.append({
                    "question_id":int(question[index]["question_id"]),
                    "answer_id":shortuuid.uuid(),
                    "model_id": args.model_id,
                    "choices":[{"index": 0, "turns": [response]}],
                    "tstamp": time.time(),
                    })            
            predictions_file = "./data/{}/model_answer/{}.jsonl".format(args.benchmark, args.model_id)
            dirname = os.path.dirname(predictions_file)
            os.makedirs(dirname,exist_ok=True)
            with open(predictions_file,'w') as f:
                for tmp_dict in results:
                    json.dump(tmp_dict,f,ensure_ascii=False)
                    f.write("\n")
            #with open(dirname+'/generation_config.json','w') as f:
                #json.dump(generation_config,f,ensure_ascii=False,indent=2)
