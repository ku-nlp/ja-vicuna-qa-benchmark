python gen_rinna_answer.py \
    --base_model "decapoda-research/llama-7b-hf"\
    --lora_model "kunishou/Japanese-Alpaca-LoRA-7b-v0"\
    --model_id japanese-alpaca-lora-7b\
    --with_prompt \
    --gpus 0 \
    --max_new_tokens 300 \
    --benchmark jp_bench \
