# Japanese Vicuna QA Benchmark

We released Japanese Vicuna QA Benchmark for measuring comprehensive capabilities of Japanese LLMs, which consists of 80 diverse questions in 10 categories (generic, coding, roleplay, writing, etc.)
You can leverage this package to evaluate the answers of your Japanese LLM models in a reference-free manner with LLM-as-a-judge.
To automate the evaluation process, we prompt strong LLMs like GPT-4 to act as judges and assess the quality of the models' responses.

To be clarified, such zero-shot QA-style evaluation might be more suitable for those LLMs that have been fine-tuned with instructions. The 80 questions are manually translated from the English Vicuna benchmark.

## Contents
- [Install](#install)
- [Evaluate a model with Japanese Vicuna QA Benchmark](#evaluate-a-model-with-japanese-vicuna-qa-benchmark)
- [Sample Outputs](#sample-outputs)
- [An Example of pairwise win-rate of three Japanese LLMs](#pairwise-win-rate-of-three-japanese-llms)
- [Supported baseline Models](#supported-baseline-models)

## Install
```
pip install -e .
```

## Evaluate a model with Japanese Vicuna QA Benchmark

#### Step 1. Generate model answers to Japanese Vicuna QA questions (noted as jp-bench).

```
python llm_judge/gen_model_answer.py --config <CONFIG-PATH>
```

Arguments & Options:
  - `<CONFIG-PATH>` is the path to a configuration file. Examples are in `configs/`.

For example:

```
python gen_model_answer.py --config configs/rinna--japanese-gpt-neox-3.6b-instruction-ppo.json
```

The answers will be saved to `data/jp_bench/model_answer`.

#### Step 2. Generate GPT-4 judgments

There are several options to use GPT-4 as a judge, such as pairwise win-rate and single-answer grading.
We show an example of the pairwise win-rate evaluation of instruction fine-tuned models (rinna-3.6b-sft-v2, rinna-3.6b-ppo, and japanese-alpaca-lora-7b) at the bottom.

```
OPENAI_API_KEY=<YOUR-KEY> python gen_judgment.py \
    --mode {single|pairwise-baseline|pairwise-all} \
    [--model-list <LIST-OF-MODEL-IDS>]
```

Arguments & Options:
- `--mode {single|pairwise-baseline|pairwise-all}` is the mode of judgment.
    - `single`: run score-based single-model grading.
    - `pairwise-baseline`: run pairwise comparison against a baseline model.
    - `pairwise-all`: run pairwise comparison between all model pairs.
- `--model-list <LIST-OF-MODEL-IDS>` is a list of model IDs to be evaluated. If not specified, all models in `data/jp_bench/model_answer` will be evaluated.

For example:

```
OPENAI_API_KEY=<YOUR-KEY> python gen_judgment.py \
    --mode pairwise-all \
    --model-list rinna-3.6b-sft-v2 rinna-3.6b-ppo japanese-alpaca-lora-7b
```

The judgments will be saved to `data/jp_bench/model_judgment/gpt-4_pair.jsonl`

#### Step 3. Show jp-bench scores

Show the scores for selected models.

```
python show_result.py \
    --mode pairwise-all \
    --model-list rinna-3.6b-sft-v2 rinna-3.6b-ppo japanese-alpaca-lora-7b
```

---

#### Pairwise comparison against a baseline (default: gpt-3.5-turbo)

The `pairwise-baseline` mode runs pairwise comparison against a baseline model.

Generate GPT-4 judgments:

```
OPENAI_API_KEY=<YOUR-KEY> python gen_judgment.py \
  --mode pairwise-baseline \
  [--model-list <LIST-OF-MODEL-IDS>] \
  [--baseline-model <name-of-baseline-model>]
```

The judgments will be saved to `data/jp_bench/model_judgment/gpt-4_pair.jsonl`

Show results:

```
python show_result.py \
  --bench-name "jp_bench" \
  --mode pairwise-baseline \
  [--model-list <LIST-OF-MODEL-IDS>]
```

#### Run GPT-4 judge with the single model

The `single` mode grades models by asking GPT-4 to grade and give a score to model's answer directly without pairwise comparison.

Generate GPT-4 judgments:

```
OPENAI_API_KEY=<YOUR-KEY> python gen_judgment.py \
  --mode single \
  [--model-list <LIST-OF-MODEL-IDS>]
```

The judgments will be saved to `data/jp_bench/model_judgment/gpt-4_single.jsonl`

Show results:

```
python show_result.py \
  --bench-name "jp_bench" \
  --mode single \
  [--model-list <LIST-OF-MODEL-IDS>]
```

## Sample Outputs

#### Question:
> 植物性タンパク源と動物性タンパク源の違いは何ですか？
#### Answer of llm-jp--llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0:
> 動物性タンパク質は、牛、羊、ヤギ、豚、魚、甲殻類、昆虫、乳製品など、さまざまな種類の生き物から得ることができる。植物性タンパク質は、大豆、ナッツ類、種子類、野菜、果物などのさまざまな植物から得ることができる。
#### Answer of llm-jp--llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0:
> 植物性タンパク源は動物性タンパク源に比べて消化・吸収が遅いため、植物性タンパク源を多く摂取すると、動物性タンパク源よりも満腹感が得られます。さらに、植物性タンパク源は、動物性タンパク源よりもグルテン、乳糖、その他の成分が少ないため、より健康的です。
#### openai--text-davinci-003:
> 植物性タンパク源は、植物から抽出されたタンパク質で構成されています。一方、動物性タンパク源は、動物から抽出されたタンパク質で構成されています。植物性タンパク源は、植物から抽出されたタンパク質であるため、植物性タンパク源は、動物性タンパク源よりも低カロリーであり、低脂肪である可能性があります。また、植物性タンパク源は、動物性タンパク源よりもビタミンやミネラルなどの有益な栄養素を多く含んでいる可能性があります。
#### Answer of rinna--japanese-gpt-neox-3.6b-instruction-ppo:
> 植物性タンパク源とは、大豆やナッツ類などの豆類、豆腐、テンペなど、植物から作られるタンパク質源です。一方、動物性タンパク源とは、牛肉や豚肉などの肉類、卵、牛乳、チーズなどの乳製品、魚介類など、主に動物から得られるタンパク質源です。
#### Answer of rinna--japanese-gpt-neox-3.6b-instruction-sft-v2:
> それは、肉を食べるか魚を食べるか、です。
#### Answer of cyberagent--calm2-7b-chat:
> 植物性タンパク源と動物性タンパク源の違いは以下の通りです。\n\n1. 供給源: 植物性タンパク源は主に植物、特に穀物から得られます。一方、動物性タンパク源は動物、特に哺乳類から得られます。\n\n2. 生物的特性: 植物性タンパク源は、植物が有する、植物の、特に植物の、植物性タンパク源の、特に植物性タンパク源の、植物性タンパク源の、特に植物性タンパク源の、特に特に特に特に特に特に特に特に特に特に


[//]: # (## Comparison )

[//]: # (![Comparison]&#40;./comparison.png&#41;  )

## Pairwise win-rate compared with GPT-3.5-davinci-003
| Model                                                    | Win | Loss | Tie | Win Rate | Loss Rate | Win Rate Adjusted |
|----------------------------------------------------------|-----|------|-----|----------|-----------|-------------------|
| llm-jp--llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0 |  22 |   48 |  10 | 0.2750   | 0.6000    | 0.33750           |
| rinna--japanese-gpt-neox-3.6b-instruction-ppo            |  10 |   61 |   9 | 0.1250   | 0.7625    | 0.18125           |
| llm-jp--llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 |   7 |   65 |   8 | 0.0875   | 0.8125    | 0.13750           |
| rinna--japanese-gpt-neox-3.6b-instruction-sft-v2         |   8 |   69 |   3 | 0.1000   | 0.8625    | 0.11875           |
| cyberagent--calm2-7b-chat                                |   5 |   67 |   8 | 0.0625   | 0.8375    | 0.11250           |

The GPT4 judgments is placed in `data/jp_bench/model_judgment/gpt-4_pair.jsonl`.

To be noticed, `pairwise-all` might become very inefficient when evaluating more LLMs, as it evaluates combinations of each two of them. In such cases, we recommend using the `pairwise-baseline` mode, allowing all models to be compared against a fixed baseline such as ChatGPT.

## Supported baseline Models
To make it more convenient for users to utilize pairwise comparisons with existing Japanese LLMs, we offer the prediction of the following four baselines in `fastchat/llm_judge/data/jp_bench/model_answer`.

- [Rinna-3.6B](https://huggingface.co/rinna/japanese-gpt-neox-3.6b)
- [Rinna-3.6B-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2)
- [Rinna-3.6B-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)
- [Japanese-Alpaca-Lora](https://huggingface.co/kunishou)

We will regularly include latest LLM baselines.

## Questions
If you have any questions and feedback, please feel free to leave questions in the `Issues' list.
