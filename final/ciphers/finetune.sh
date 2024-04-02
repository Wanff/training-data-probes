export CUDA_VISIBLE_DEVICES=1

python finetune_manual.py \
    --path "/home/ubuntu/gld/train-data-probes/data/70m/ciphers" \
    --model_name "EleutherAI/pythia-70m" \
    --dataset_name "rotated_7" \
    --output_name "rotated_7_model" \
    --seed 0 \
    --lr 5e-5 \
    --batch_size 16 \
    --num_epochs 2 \

# python finetune.py \
#     --path "/home/ubuntu/gld/train-data-probes/data/70m/ciphers" \
#     --model_name "EleutherAI/pythia-70m" \
#     --dataset_name "rotated_3" \
#     --output_name "rotated_3_model" \
#     --seed 0 \
#     --lr 5e-5 \
#     --batch_size 8 \
#     --num_epochs 2 \
