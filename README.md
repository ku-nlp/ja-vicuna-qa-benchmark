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

```bash
pip install -e .
```

## Evaluate a model with Japanese Vicuna QA Benchmark

#### Step 1. Generate model answers to Japanese Vicuna QA questions (noted as jp-bench).

```bash
python llm_judge/gen_model_answer.py --config <CONFIG-PATH>
```

Arguments & Options:
  - `<CONFIG-PATH>` is the path to a configuration file. Examples are in `configs/`.

For example:

```bash
python llm_judge/gen_model_answer.py --config configs/rinna--japanese-gpt-neox-3.6b-instruction-ppo.json
```

#### Step 2. Generate GPT-4 judgments

There are several options to use GPT-4 as a judge, such as pairwise win-rate and single-answer grading.

```bash
OPENAI_API_KEY=<YOUR-KEY> python llm_judge/gen_judgment.py \
    --mode {single|pairwise-baseline|pairwise-all} \
    [--baseline-model <BASELINE-MODEL-ID>] \
    [--model-list <LIST-OF-MODEL-IDS>]
```

Arguments & Options:
- `--mode {single|pairwise-baseline|pairwise-all}` is the mode of judgment.
    - `pairwise-baseline`: run pairwise comparison against a baseline model.
    - `pairwise-all`: run pairwise comparison between all model pairs.
    - `single`: run score-based single-model grading.
- `--baseline-model <BASELINE-MODEL-ID>` is the model ID of the baseline model. This option is only available in `pairwise-baseline` mode. If not specified, the baseline model is set to `text-davinci-003`.
- `--model-list <LIST-OF-MODEL-IDS>` is a list of model IDs to be evaluated. If not specified, all models in `data/jp_bench/model_answer` will be evaluated.

**Mode: `pairwise-baseline`**

This mode runs pairwise comparison against a baseline model.
By default, the baseline model is set to `text-davinci-003`.
For example:

```bash
OPENAI_API_KEY=<YOUR-KEY> python llm_judge/gen_judgment.py \
    --mode pairwise-baseline \
    --model-list rinna--japanese-gpt-neox-3.6b-instruction-ppo
```

To show the scores:

```bash
python llm_judge/show_result.py \
    --mode pairwise-baseline \
    --model-list rinna--japanese-gpt-neox-3.6b-instruction-ppo
```

**Mode: `pairwise-all`**

This mode runs pairwise comparison between all model pairs.
For example:

```bash
OPENAI_API_KEY=<YOUR-KEY> python llm_judge/gen_judgment.py \
    --mode pairwise-all
```

To show the scores:

```bash
python llm_judge/show_result.py \
    --mode pairwise-all
```

To be noticed, the `pairwise-all` mode might become very inefficient when evaluating many LLMs, as it evaluates combinations of each two of them.
In such cases, we recommend using the `pairwise-baseline` mode, allowing all models to be compared against a fixed baseline such as ChatGPT.

**Mode: `single`**

This mode runs score-based single-model grading.
For example:

```bash
python llm_judge/gen_judgment.py \
    --mode single \
    --model-list rinna--japanese-gpt-neox-3.6b-instruction-ppo
```

To show the scores:

```bash
python llm_judge/show_result.py \
    --mode single \
    --model-list rinna--japanese-gpt-neox-3.6b-instruction-ppo
```

## Sample Outputs

Question: 植物性タンパク源と動物性タンパク源の違いは何ですか？

Model outputs:

| Model                                                    | Answer                                                                                                                                                                                                                                                                                                                                 |
|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| openai--text-davinci-003                                 | 植物性タンパク源は、植物から抽出されたタンパク質で構成されています。一方、動物性タンパク源は、動物から抽出されたタンパク質で構成されています。植物性タンパク源は、植物から抽出されたタンパク質であるため、植物性タンパク源は、動物性タンパク源よりも低カロリーであり、低脂肪である可能性があります。また、植物性タンパク源は、動物性タンパク源よりもビタミンやミネラルなどの有益な栄養素を多く含んでいる可能性があります。                                                                                                                          |
| llm-jp--llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 | 動物性タンパク質は、牛、羊、ヤギ、豚、魚、甲殻類、昆虫、乳製品など、さまざまな種類の生き物から得ることができる。植物性タンパク質は、大豆、ナッツ類、種子類、野菜、果物などのさまざまな植物から得ることができる。                                                                                                                                                                                                                               |
| llm-jp--llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0 | 植物性タンパク源は動物性タンパク源に比べて消化・吸収が遅いため、植物性タンパク源を多く摂取すると、動物性タンパク源よりも満腹感が得られます。さらに、植物性タンパク源は、動物性タンパク源よりもグルテン、乳糖、その他の成分が少ないため、より健康的です。                                                                                                                                                                                                           |
| rinna--japanese-gpt-neox-3.6b-instruction-ppo            | 植物性タンパク源とは、大豆やナッツ類などの豆類、豆腐、テンペなど、植物から作られるタンパク質源です。一方、動物性タンパク源とは、牛肉や豚肉などの肉類、卵、牛乳、チーズなどの乳製品、魚介類など、主に動物から得られるタンパク質源です。                                                                                                                                                                                                                    |
| rinna--japanese-gpt-neox-3.6b-instruction-sft-v2         | それは、肉を食べるか魚を食べるか、です。                                                                                                                                                                                                                                                                                                                   |
| cyberagent--calm2-7b-chat                                | 植物性タンパク源と動物性タンパク源の違いは以下の通りです。\n\n1. 供給源: 植物性タンパク源は主に植物、特に穀物から得られます。一方、動物性タンパク源は動物、特に哺乳類から得られます。\n\n2. 生物的特性: 植物性タンパク源は、植物が有する、植物の、特に植物の、植物性タンパク源の、特に植物性タンパク源の、植物性タンパク源の、特に植物性タンパク源の、特に特に特に特に特に特に特に特に特に特に                                                                                                                               |

## Pairwise win-rate compared with GPT-3.5-davinci-003

| Model                                                    | Win | Loss | Tie | Win Rate | Loss Rate | Win Rate Adjusted |
|----------------------------------------------------------|-----|------|-----|----------|-----------|-------------------|
| llm-jp--llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0 |  22 |   48 |  10 | 0.2750   | 0.6000    | 0.33750           |
| rinna--japanese-gpt-neox-3.6b-instruction-ppo            |  10 |   61 |   9 | 0.1250   | 0.7625    | 0.18125           |
| llm-jp--llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 |   7 |   65 |   8 | 0.0875   | 0.8125    | 0.13750           |
| rinna--japanese-gpt-neox-3.6b-instruction-sft-v2         |   8 |   69 |   3 | 0.1000   | 0.8625    | 0.11875           |
| cyberagent--calm2-7b-chat                                |   5 |   67 |   8 | 0.0625   | 0.8375    | 0.11250           |

## Supported baseline Models

To make it more convenient for users to utilize pairwise comparisons with existing Japanese LLMs, we offer the prediction of the following four baselines in `data/jp_bench/model_answer`.

- [llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0)
- [llm-jp/llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0)
- [rinna/japanese-gpt-neox-3.6b-instruction-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)
- [rinna/japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2)
- [cyberagent/calm2-7b-chat](https://huggingface.co/cyberagent/calm2-7b-chat)

## Questions

If you have any questions and feedback, please feel free to leave questions in the `Issues' list.
