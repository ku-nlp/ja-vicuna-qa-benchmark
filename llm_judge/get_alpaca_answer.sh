python gen_model_answer.py \
    --base_model "decapoda-research/llama-7b-hf"\
    --lora_model "/home/sun/LLaMA-Efficient-Tuning/alpaca_jp-14600"\
    --model_id alpaca-jp-selfinstruction-14600\
    --gpus 0 \
    --max_new_tokens 400 \
    --benchmark jp_bench \
