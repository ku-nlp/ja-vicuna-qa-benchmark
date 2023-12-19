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
    [--model-list <LIST-OF-MODEL-IDS>] \
    [--yes] \
    [--wandb]
```

Arguments & Options:
- `--mode {single|pairwise-baseline|pairwise-all}` is the mode of judgment.
    - `pairwise-baseline`: run pairwise comparison against a baseline model. This mode will be used by default.
    - `pairwise-all`: run pairwise comparison between all model pairs.
    - `single`: run score-based single-model grading.
- `--baseline-model <BASELINE-MODEL-ID>` is the model ID of the baseline model. This option is only available in `pairwise-baseline` mode. If not specified, the baseline model is set to `text-davinci-003`.
- `--model-list <LIST-OF-MODEL-IDS>` is a list of model IDs to be evaluated. If not specified, all models in `data/jp_bench/model_answer` will be evaluated.
- `--yes` is a flag to skip the confirmation prompt.
- `--wandb` is a flag to enable logging to W&B. You can upload the results later to W&B by running `upload_result.py`, as described in the next section.

**Mode: `pairwise-baseline` (Default)**

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

#### Step 3. Upload the results to W&B (Optional)

If you want to upload the results to W&B, you can run the following command:

```bash
WANDB_ENTITY=<USER-NAME or ORGANIZATION-NAME> WANDB_PROJECT=<PROJECT-NAME> python llm_judge/upload_result.py \
    --mode {single|pairwise-baseline|pairwise-all} \
    [--baseline-model <BASELINE-MODEL-ID>] \
    [--model-list <LIST-OF-MODEL-IDS>]
```

By default, the entity is configured to use your username, and the project name is set to `ja-vicuna-qa-benchmark-dev-<VERSION>`.

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

| Model                                                    | Win Rate | Loss Rate | Win Rate Adjusted |
|----------------------------------------------------------|----------|-----------|-------------------|
| llm-jp--llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0 | 28.7     | 62.5      | 33.1              |
| rinna--japanese-gpt-neox-3.6b-instruction-ppo            | 13.8     | 13.8      | 18.8              |
| rinna--japanese-gpt-neox-3.6b-instruction-sft-v2         | 8.8      | 82.5      | 13.1              |
| cyberagent--calm2-7b-chat                                | 6.2      | 81.2      | 12.5              |
| llm-jp--llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 | 10.0     | 87.5      | 11.2              |

## Supported baseline Models

To make it more convenient for users to utilize pairwise comparisons with existing Japanese LLMs, we offer the prediction of the following four baselines in `data/jp_bench/model_answer`.

- [llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0)
- [llm-jp/llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0)
- [rinna/japanese-gpt-neox-3.6b-instruction-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)
- [rinna/japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2)
- [cyberagent/calm2-7b-chat](https://huggingface.co/cyberagent/calm2-7b-chat)

## Questions

If you have any questions and feedback, please feel free to leave questions in the `Issues' list.
