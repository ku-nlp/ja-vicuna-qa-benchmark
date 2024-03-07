# Japanese Vicuna QA Benchmark

This repository contains code for Japanese Vicuna QA Benchmark, described by the paper: [Rapidly Developing High-quality Instruction Data and Evaluation Benchmark for Large Language Models with Minimal Human Effort: A Case Study on Japanese](https://arxiv.org/pdf/2010.12812.pdf).



We released Japanese Vicuna QA Benchmark for measuring comprehensive capabilities of Japanese LLMs, which consists of 80 diverse questions in 10 categories (generic, coding, roleplay, writing, etc.)
You can leverage this package to evaluate the answers of your Japanese LLM models in a reference-free manner with LLM-as-a-judge.
To automate the evaluation process, we prompt strong LLMs like GPT-4 to act as judges and assess the quality of the models' responses.

To be clarified, such zero-shot QA-style evaluation might be more suitable for those LLMs that have been fine-tuned with instructions. The 80 questions are manually translated from the English Vicuna benchmark.

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

## Pairwise win-rate compared with GPT-3.5 (text-davinci-003)

See the [leaderboard](http://wandb.me/llm-jp-vicunaleaderboard) (in Japanese).

## Supported baseline Models

To make it more convenient for users to utilize pairwise comparisons with existing Japanese LLMs, we offer the prediction of the following four baselines in `data/jp_bench/model_answer`.

- [llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0)
- [llm-jp/llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0](https://huggingface.co/llm-jp/llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0)
- [rinna/japanese-gpt-neox-3.6b-instruction-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)
- [rinna/japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2)
- [cyberagent/calm2-7b-chat](https://huggingface.co/cyberagent/calm2-7b-chat)
- [tokyotech-llm/Swallow-70b-instruct-hf](https://huggingface.co/tokyotech-llm/Swallow-70b-instruct-hf)

## Questions

If you have any questions and feedback, please feel free to leave questions in the `Issues' list.

## Citation
If you use our code in your research, please cite our work:
```bibtex
@inproceedings{sun2024rapidly,
   title={Rapidly Developing High-quality Instruction Data and Evaluation Benchmark for Large Language Models with Minimal Human Effort: A Case Study on Japanese},
   author={Sun, Yikun and Wan, Zhen and Ueda, Nobuhiro and Yahata, Sakiko and Cheng, Fei and Chu, Chenhui and Kurohashi, Sadao},
   booktitle={The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
   year={2024}
}
```

