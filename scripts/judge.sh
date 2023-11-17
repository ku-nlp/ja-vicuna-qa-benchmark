python -B llm_judge/gen_judgment.py \
    --bench-name "jp_bench" \
    --mode pairwise-baseline  \
    --baseline-model gpt-3.5-davinci \
    --model-list llm-jp-13b-lora-sft-gpt4-self-instruct\
    --parallel 1
