#!/bin/bash

SEEDS=(0 1 2 3)
MODES=('hyp')
LR=(1.)

for k_lr in "${LR[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for mode in "${MODES[@]}"; do
            echo "Running ${mode} attn with k_lr = ${k_lr} and seed = ${seed}"
            OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
                train_gpt2.py \
                --data_path "data/shakespeare_char" \
                --gen_prompt "THIBAULT: " \
                --device_batch_size 64 \
                --batch_size 64 \
                --num_iterations 6001 \
                --gen_every 500 \
                --train_loss_every 50 \
                --val_loss_every 50 \
                --n_heads 8 \
                --n_layers 8 \
                --head_dim 8 \
                --sequence_length 256 \
                --attn_mode "$mode" \
                --head_mode "euc"\
                --curvature 1. \
                --k_lr "$k_lr" \
                --seed "$seed" \
                > logs/${NAME}_${mode}_lr${k_lr}_${seed}.txt 2>&1
        done
    done
done
