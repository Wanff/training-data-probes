# MAIN PYTHIA-70M RESULTS

# python3 get_activations.py \
#     --model_name "EleutherAI/pythia-70m" \
#     --dataset "pile" \
#     --save_path "/home/ubuntu/gld/train-data-probes/data/70m/pile" \
#     --act_types "mlp" "attn" "attn_v" "attn_q" "attn_k" \
#     --N_PROMPTS 5000 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging

# python3 get_activations.py \
#     --model_name "EleutherAI/pythia-70m" \
#     --dataset "pythia-evals" \
#     --save_path "/home/ubuntu/gld/train-data-probes/data/70m/pythia-evals" \
#     --act_types "mlp" "attn" "attn_v" "attn_q" "attn_k" \
#     --N_PROMPTS 5000 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging

# generalization dataset 
python3 get_activations.py \
    --model_name "EleutherAI/pythia-70m" \
    --dataset "pythia-evals-12b" \
    --save_path "/home/ubuntu/gld/train-data-probes/data/70m/pythia-evals-12b" \
    --act_types "mlp" "attn" "attn_v" "attn_q" "attn_k" \
    --N_PROMPTS 5000 \
    --save_every 100 \
    --return_prompt_acts \
    --logging

# # # run with pythia 6.9b
# python3 get_activations.py \
#     --model_name "EleutherAI/pythia-1b" \
#     --dataset "pile-test" \
#     --save_path "/home/ubuntu/gld/train-data-probes/data/1b" \
#     --N_PROMPTS 10000 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging