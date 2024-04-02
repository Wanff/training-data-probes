export CUDA_VISIBLE_DEVICES='2,3'

python finetune.py \
    --path "/home/ubuntu/gld/train-data-probes/data/1b/ciphers" \
    --model_name "EleutherAI/pythia-1b" \
    --dataset_name "rotated_7" \
    --output_name "rotated_7_model" \
    --seed 0 \
    --lr 5e-5 \
    --batch_size 8 \
    --num_epochs 2 \
