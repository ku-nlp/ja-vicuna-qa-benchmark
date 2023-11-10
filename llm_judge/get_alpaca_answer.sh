python gen_model_answer.py \
    --base_model /model/checkpoint_HF/13B/ds_gpt_v101_fattn_nfs_0825_refined-data-gpt_13B_refined_gpu96_node12_lr0.00008533_gbs1536_mbs1_nwk2_zero1_pp8/global_step96657\
    --tokenizer_path llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 \
    --lora_model /model/wan/llm-jp-13b-refined-52k/results \
    --model_id llm-jp-13b-lora-sft-gpt4-self-instruct\
    --with_prompt \
    --gpus 0 \
    --max_new_tokens 400 \
    --benchmark jp_bench \
