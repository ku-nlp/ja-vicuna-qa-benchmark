OPENAI_API_KEY=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 python -B gen_judgment.py \
    --bench-name "jp_bench" \
    --mode pairwise-all \
    --model-list rinna-3.6b rinna-3.6b-sft-v2 rinna-3.6b-ppo\
    --parallel 2
