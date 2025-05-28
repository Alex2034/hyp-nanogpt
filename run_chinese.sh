#!/bin/bash
# SEEDS=(0)
# MODES=('hyp')

SEEDS=(0 1 2 3)
MODES=('hyp' 'euc')

NAME='cn'

for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 \
            train_gpt2.py \
            --data_path "data/cn_wiki" \
            --gen_prompt "Lily " \
            --device_batch_size 140 \
            --batch_size 140 \
            --num_iterations 2001 \
            --gen_every 500 \
            --save_every 500 \
            --train_loss_every 50 \
            --val_loss_every 50 \
            --n_heads 12 \
            --n_layers 12 \
            --head_dim 16 \
            --sequence_length 512 \
            --attn_mode "$mode" \
            --head_mode "euc"\
            --curvature 1. \
            --k_lr 10. \
            --seed "$seed" \
            --print_multiplier 5 \
            > new_logs/${NAME}_${mode}_run_${seed}.txt 2>&1
    done
done
