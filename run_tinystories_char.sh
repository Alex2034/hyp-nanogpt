#!/bin/bash
# SEEDS=(0)
# MODES=('hyp')

SEEDS=(0 1 2 3)
MODES=('hyp' 'euc')
LR=(1. 10.)

NAME='tsc'

for k_lr in "${LR[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for mode in "${MODES[@]}"; do
            OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 \
                train_gpt2.py \
                --data_path "data/tinystories_char" \
                --gen_prompt "Lily " \
                --device_batch_size 192 \
                --batch_size 192 \
                --num_iterations 5001 \
                --gen_every 500 \
                --gen_length 500 \
                --train_loss_every 50 \
                --val_loss_every 50 \
                --n_heads 12 \
                --n_layers 12 \
                --head_dim 8 \
                --sequence_length 512 \
                --attn_mode "$mode" \
                --head_mode "euc"\
                --curvature 1. \
                --k_lr "$k_lr" \
                --seed "$seed" \
                > new_logs/${NAME}_${mode}_run_${seed}.txt 2>&1
        done
    done
done
