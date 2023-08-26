# Japanese Vicuna QA Benchmark

 We released Japanese Vicuna QA Benchmark for measuring comprehensive capabilities of Japanese LLMs, which consists of 80 diverse questions in 10 categories (generic, coding, roleplay, writing, etc.) 
You can leverage this package to evaluate the answers of your Japanese LLM models in a reference-free manner with LLM-as-a-judge.
To automate the evaluation process, we prompt strong LLMs like GPT-4 to act as judges and assess the quality of the models' responses.

To be clarified, such zero-shot QA-style evaluation might be more suitable for those LLMs that have been fine-tuned with instructions. The 80 questions are manually translated from the English Vicuna benchmark.

## Contents
- [Install](#install)
- [Review Pre-Generated Model Answers and Judgments](#review-pre-generated-model-answers-and-judgments)
## Install
```
git clone https://github.com/hitoshizuku7/LLM_Judge_ku.git
cd LLM_Judge_ku
pip install -e .
pip install openai anthropic ray
cd fastchat/llm_judge
```


### Evaluate a model on Japanese Vicuna QA Benchmark (noted as jp-bench).

#### Step 1. Generate model answers to jp-bench questions
```
python gen_model_answer.py \
--base_model [MODEL-PATH] \
--lora_model [LORA-PATH] \
--model-id [MODEL-ID] \
--with_prompt \
--gpus [GPU_Num] \
--max_new_tokens [NUM of NEW TOKENS] \
--benchmark jp_bench
```
Arguments:
  - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
  - `[LORA-PATH]` is the path to the lora weights if needed.
  - `[MODEL-ID]` is a name you give to the model.
  - `[GPU_Num]` denotes which GPU you decide to use


e.g.,
```
python gen_model_answer.py \
--model-path rinna/japanese-gpt-neox-3.6b-instruction-ppo \
--model-id rinna-3.6b-ppo \
--with_prompt \
--gpus 0 \
--max_new_tokens 2048 \
--benchmark jp_bench
```
The answers will be saved to `data/jp_bench/model_answer/[MODEL-ID].jsonl`.

You can also specify `--num-gpus-per-model` for model parallelism (needed for large 65B models) and `--num-gpus-total` to parallelize answer generation with multiple GPUs.

#### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise win-rate and single-answer grading. We show an example of the pairwise win-rate evaluation of instruction fine-tuned models (rinna-3.6b-sft-v2, rinna-3.6b-ppo, and japanese-alpaca-lora-7b) at the bottom.

```
OPENAI_API_KEY=[YOUR-KEY] python -B gen_judgment.py \
--bench-name "jp_bench" \
--mode [pairwise-all, single, pairwise-baseline] \
--model-list [LIST-OF-MODEL-ID] \
--parallel [num-concurrent-api-call]
```

e.g.,
```
OPENAI_API_KEY=[YOUR-KEY] python -B gen_judgment.py \
--bench-name "jp_bench" \
--mode pairwise-all \
--model-list rinna-3.6b-sft-v2 rinna-3.6b-ppo japanese-alpaca-lora-7b \
--parallel 2
```
`pairwise-all`: run pairwise comparison between all model pairs.
The judgments will be saved to `data/jp_bench/model_judgment/gpt-4_pair.jsonl`

#### Step 3. Show jp-bench scores

- Show the scores for selected models
  ```
  python show_result.py \
  --bench-name "jp_bench" \
  --mode pairwise-all \
  --model-list rinna-3.6b-sft-v2 rinna-3.6b-ppo japanese-alpaca-lora-7b
  ```

---

### Other grading options
For GPT-4 judgments, besides score-based single-answer grading, we also support two additional grading options based on win rates:
- `pariwise-baseline`: run pairwise comparison against a baseline model.
- `single`: run score-based single-model grading.

#### Option 2: pairwise comparison against a baseline (default: gpt-3.5-turbo)

- Generate GPT-4 judgments
```
OPENAI_API_KEY=[YOUR-KEY] python -B gen_judgment.py \
--bench-name "jp_bench" \
--mode pairwise-baseline \
--model-list [LIST-OF-MODEL-ID] \
--parallel [num-concurrent-api-call]
```
The judgments will be saved to `data/jp_bench/model_judgment/gpt-4_pair.jsonl`

- Show results
```
python show_result.py \
--bench-name "jp_bench" \
--mode pairwise-baseline \
--model-list [LIST-OF-MODEL-ID] \
```

#### Option 3: Run GPT-4 judge with the single model

The `single` mode grades models by asking GPT-4 to grade and give a score to model's answer directly without pairwise comparison.

```
OPENAI_API_KEY=[YOUR-KEY] python -B gen_judgment.py \
--bench-name "jp_bench" \
--mode single \
--model-list [LIST-OF-MODEL-ID] \
--parallel [num-concurrent-api-call]
```

```
python show_result.py \
--bench-name "jp_bench" \
--mode single
--model-list [LIST-OF-MODEL-ID] \
```


## Sample Outputs

#### Question: 
植物性タンパク源と動物性タンパク源の違いは何ですか？
#### Answer of rinna-3.6b: 
栄養素の含有量に大きな違いがあります。
#### Answer of rinna-3.6b-sft-v2: 
植物性タンパク源は肉や魚など、いくつかの種類の食品を指します。一方、動物性タンパク源には卵、乳製品、大豆、ナッツ、種などが含まれます。
#### Answer of rinna-3.6b-ppo: 
植物性タンパク源とは、主に大豆や豆腐などの大豆製品を生産する植物によって作られるタンパク質で、卵、牛乳、肉などの動物性タンパク源よりも栄養価が高く、低カロリーです。一方、動物性タンパク源には、牛肉、豚肉、鶏肉などが含まれます。
#### Answer of japanese-alpaca-lora-7b:
動物タンポク源（ATP）は、動物によって生産されるエネルギー源である。ATPは、細胞におけるATP認識システムの活性化により生成されています。動物のATP源は、遺伝的に組み込まれたATP生成機構であり、これは、ATPを生成するために使用され、経験的にATPの量を増加させることができる。

[//]: # (## Comparison )

[//]: # (![Comparison]&#40;./comparison.png&#41;  )

## Pairwise win-rate of three Japanese LLMs
| Model                   | win | loss | tie |  win_rate | loss_rate | win_rate_adjusted |
|:------------------------|----:|-----:|----:|----------:|----------:|------------------:|
| rinna-3.6b-ppo          |  87 |   25 |  52 |  0.530488 |  0.152439 |          0.689024 |
| rinna-3.6b-sft-v2       |  70 |   35 |  58 |  0.429448 |  0.214724 |          0.607362 |
| japanese-alpaca-lora-7b |  15 |  112 |  38 |  0.090909 |  0.678788 |          0.206061 |

The GPT4 judgments is placed in `data/jp_bench/model_judgment/gpt-4_pair.jsonl`. 

To be noticed, `pairwise-all` might become very inefficient when evaluating more LLMs, as it evaluates combinations of each two of them. In such cases, we recommend using the `pairwise-baseline` mode, allowing all models to be compared against a fixed baseline such as ChatGPT.

## Supported baseline Models
To make it more convenient for users to utilize pairwise comparisons with existing Japanese LLMs, we offer the prediction of the following four baselines in `fastchat/llm_judge/data/jp_bench/model_answer`. 

[Rinna-3.6B](https://huggingface.co/rinna/japanese-gpt-neox-3.6b)

[Rinna-3.6B-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2)

[Rinna-3.6B-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)

[Japanese-Alpaca-Lora](https://huggingface.co/kunishou)

We will regularly include latest LLM baselines.

## Questions
If you have any questions and feedback, please feel free to send a mail to feicheng@i.kyoto-u.ac.jp or leave questions in the `Issues' list.
