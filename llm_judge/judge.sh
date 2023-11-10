OPENAI_API_KEY=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 python -B gen_judgment.py \
    --bench-name "jp_bench" \
    --mode pairwise-baseline  \
    --baseline-model gpt-3.5-davinci \
    --model-list llm-jp-13b-lora-sft-gpt4-self-instruct\
    --parallel 1
