#!/bin/bash

SEEDS=(0)
MODES=('hyp')
# SEEDS=(0 1 2 3 4)
LR=(1. 0.)


for k_lr in "${LR[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for mode in "${MODES[@]}"; do
            OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
                train_gpt2.py \
                --data_path "data/finewebedu" \
                --gen_prompt "What is " \
                --device_batch_size 42 \
                --batch_size 84 \
                --num_iterations 51 \
                --gen_every 2000 \
                --train_loss_every 10 \
                --val_loss_every 10 \
                --n_heads 12 \
                --n_layers 12 \
                --head_dim 16 \
                --sequence_length 1024 \
                --attn_mode "$mode" \
                --head_mode "euc"\
                --curvature 1. \
                --k_lr "$k_lr" \
                --seed "$seed" \
                > new_logs/fwe_${mode}_lr${k_lr}_run_${seed}.txt 2>&1
        done
    done
done