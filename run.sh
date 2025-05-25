#!/bin/bash
SEEDS=(0)
# SEEDS=(0 1 2 3 5)
for seed in "${SEEDS[@]}"; do
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
        train_gpt2_min.py \
        --data_path "data/cn_wiki" \
        --gen_prompt "Chinese " \
        --device_batch_size 16 \
        --batch_size 16 \
        --num_iterations 1001 \
        --gen_every 200 \
        --train_loss_every 20 \
        --val_loss_every 20 \
        --n_heads 12 \
        --n_layers 6 \
        --head_dim 16 \
        --sequence_length 256 \
        --attn_mode "euc" \
        --head_mode "euc" \
        --curvature 1. \
        --k_lr 0. \
        --seed "$seed" \
        > logs/last${seed}.txt 2>&1
done