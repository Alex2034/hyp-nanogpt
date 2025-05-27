#!/bin/bash
# SEEDS=(0)
MODES=('hyp')

SEEDS=(1 2 3 5)
# MODES=('euc' 'hyp')

NAME='tao'

for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
            train_gpt2.py \
            --data_path "data/taoteching" \
            --gen_prompt "Ch. 1. " \
            --device_batch_size 16 \
            --batch_size 16 \
            --num_iterations 4001 \
            --gen_every 200 \
            --train_loss_every 20 \
            --val_loss_every 20 \
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
