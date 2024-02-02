# nohup python3 get_activations.py \
#     --model_name "EleutherAI/pythia-12b" \
#     --save_path "data/12b-last500" \
#     --N_PROMPTS 5000 \
#     --save_every 500 \
#     --check_if_memmed \
#     --return_prompt_acts \
#     --logging &> data/12b-last500/log.out &

#* second run for mlp
# nohup python3 get_activations.py \
#     --model_name "EleutherAI/pythia-12b" \
#     --save_path "data/12b" \
#     --act_types "mlp" "attn" \
#     --N_PROMPTS 500 \
#     --save_every 100 \
#     --logging &> data/12b/mlpattn_log.out &

# #* final pythia 12b run for attn/mlp
# nohup python3 get_activations.py \
#     --model_name "EleutherAI/pythia-12b" \
#     --save_path "../data/12b" \
#     --act_types "mlp" "attn" "attn_v" \
#     --N_PROMPTS 10000 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging &> ../data/12b/final_attn_run.out &

#* llama run
# nohup python3 get_activations.py \
#     --model_name "meta-llama/Llama-2-7b-hf" \
#     --save_path "../data/llama-2-7b" \
#     --N_PROMPTS 10000 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging &> ../data/llama-2-7b/log.out &

#* llama more run
# nohup python3 get_activations.py \
#     --model_name "meta-llama/Llama-2-7b-hf" \
#     --save_path "../data/llama-2-7b" \
#     --N_PROMPTS 50000 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging &> ../data/llama-2-7b/more_mem_log.out &

#* llama more run negative
# nohup python3 get_activations.py \
#     --model_name "meta-llama/Llama-2-7b-hf" \
#     --save_path "../data/llama-2-7b" \
#     --N_PROMPTS 2500 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging &> ../data/llama-2-7b/more_unmem_log.out &

#* llama run, mlp attn
# nohup python3 get_activations.py \
#     --model_name "meta-llama/Llama-2-7b-hf" \
#     --save_path "../data/llama-2-7b" \
#     --act_types "mlp" "attn" "attn_v" "attn_q" "attn_k" \
#     --N_PROMPTS 2000 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging &> ../data/llama-2-7b/mlp_attn_preg_3500_to_4500.out &

# nohup python3 get_activations.py \
#     --model_name "EleutherAI/pythia-160m" \
#     --save_path "data/160m" \
#     --N_PROMPTS 20 \
#     --save_every 5 \
#     --check_if_memmed \
#     --return_prompt_acts \
#     --logging &> data/160m/log.out &

# run with pythia 6.9b
nohup python3 get_activations.py \
    --model_name "EleutherAI/pythia-6.9b" \
    --save_path "/home/ubuntu/gld/train-data-probes/data/6.9b2" \
    --act_types "mlp" "attn" "attn_v" "attn_q" "attn_k" \
    --N_PROMPTS 50000 \
    --save_every 100 \
    --return_prompt_acts \
    --logging &> /home/ubuntu/gld/train-data-probes/data/6.9b2/log.out &

# nohup python3 get_activations.py \
#     --model_name "EleutherAI/pythia-6.9b" \
#     --save_path "/home/ubuntu/gld/train-data-probes/data/6.9b-on-llama" \
#     --act_types "mlp" "attn" "attn_v" "attn_q" "attn_k" \
#     --N_PROMPTS 10000 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging &> /home/ubuntu/gld/train-data-probes/data/6.9b-on-llama/log.out &