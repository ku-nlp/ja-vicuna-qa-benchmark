OPENAI_API_KEY=[OpenAI key] python -B gen_judgment.py \
    --bench-name "jp_bench" \
    --mode pairwise-all \
    --model-list rinna-3.6b rinna-3.6b-sft-v2 rinna-3.6b-ppo\
    --parallel 2
