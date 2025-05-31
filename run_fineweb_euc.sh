#!/bin/bash

# SEEDS=(0)
MODES=('euc')
SEEDS=(0)
LR=(1.)


for k_lr in "${LR[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for mode in "${MODES[@]}"; do
            OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 \
                train_gpt2.py \
                --data_path "data/finewebedu" \
                --gen_prompt "The " \
                --device_batch_size 25 \
                --batch_size 50 \
                --num_iterations 20001 \
                --save_every 2000 \
                --gen_every 2000 \
                --gen_length 200 \
                --train_loss_every 50 \
                --val_loss_every 50 \
                --n_heads 16 \
                --n_layers 16 \
                --head_dim 16 \
                --sequence_length 1024 \
                --attn_mode "$mode" \
                --head_mode "euc"\
                --curvature 1. \
                --k_lr "$k_lr" \
                --seed "$seed" \
                > new_logs/fwe_${mode}_run_${seed}.txt 2>&1
        done
    done
done