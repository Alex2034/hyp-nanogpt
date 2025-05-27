#!/bin/bash

# SEEDS=(0)
MODES=('hyp')
SEEDS=(0 1 2 3 4)
for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
            train_gpt2_min.py \
            --data_path "data/finewebedu" \
            --gen_prompt "What is " \
            --device_batch_size 32 \
            --batch_size 64 \
            --num_iterations 10001 \
            --gen_every 2000 \
            --train_loss_every 50 \
            --val_loss_every 50 \
            --n_heads 12 \
            --n_layers 12 \
            --head_dim 16 \
            --sequence_length 1024 \
            --attn_mode "$mode" \
            --head_mode "euc"\
            --curvature 1. \
            --k_lr 0. \
            --seed "$seed" \
            > new_logs/fwe_${mode}_run_${seed}.txt 2>&1
    done
done