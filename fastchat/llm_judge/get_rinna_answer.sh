python gen_model_answer.py \
    --base_model "rinna/japanese-gpt-neox-3.6b-instruction-ppo"\
    --model_id rinna-3.6b-ppo\
    --with_prompt \
    --gpus 0 \
    --max_new_tokens 2048 \
    --benchmark jp_bench \
