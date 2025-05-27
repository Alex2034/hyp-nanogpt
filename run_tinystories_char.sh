#!/bin/bash
SEEDS=(0)
MODES=('hyp')

# SEEDS=(0 1 2 3 4)
# MODES=('euc' 'hyp')

NAME='tsc'

for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 \
            train_gpt2.py \
            --data_path "data/tinystories_char" \
            --gen_prompt "Lily " \
            --device_batch_size 128 \
            --batch_size 128 \
            --num_iterations 5001 \
            --gen_every 500 \
            --train_loss_every 20 \
            --val_loss_every 20 \
            --n_heads 8 \
            --n_layers 8 \
            --head_dim 16 \
            --sequence_length 512 \
            --attn_mode "$mode" \
            --head_mode "euc"\
            --curvature 1. \
            --k_lr 1. \
            --seed "$seed" \
            > new_logs/${NAME}_${mode}_run_${seed}.txt 2>&1
    done
done
