nohup python3 get_activations.py \
    --model_name "EleutherAI/pythia-12b" \
    --save_path "data/12b-last500" \
    --N_PROMPTS 5000 \
    --save_every 500 \
    --check_if_memmed \
    --return_prompt_acts \
    --logging &> data/12b-last500/log.out &

# nohup python3 get_activations.py \
#     --model_name "EleutherAI/pythia-160m" \
#     --save_path "data/160m" \
#     --N_PROMPTS 20 \
#     --save_every 5 \
#     --check_if_memmed \
#     --return_prompt_acts \
#     --logging &> data/160m/log.out &