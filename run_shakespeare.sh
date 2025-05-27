#!/bin/bash
# SEEDS=(0)
# MODES=('hyp')

SEEDS=(0 1 2 3 4)
MODES=('euc' 'hyp')

NAME='sh'

for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
            train_gpt2.py \
            --data_path "data/shakespeare_char" \
            --gen_prompt "THIBAULT: " \
            --device_batch_size 64 \
            --batch_size 64 \
            --num_iterations 4001 \
            --gen_every 500 \
            --train_loss_every 50 \
            --val_loss_every 50 \
            --n_heads 6 \
            --n_layers 6 \
            --head_dim 16 \
            --sequence_length 256 \
            --attn_mode "$mode" \
            --head_mode "euc"\
            --curvature 1. \
            --k_lr 1. \
            --seed "$seed" \
            > new_logs/${NAME}_${mode}_run_${seed}.txt 2>&1
    done
done
