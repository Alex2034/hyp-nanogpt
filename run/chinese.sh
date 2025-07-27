#!/bin/bash
# SEEDS=(0)
# MODES=('hyp')

SEEDS=(0 1 2 3)
MODES=('hyp' 'euc')
NORM=('power' 'exp' 'learnable')

NAME='cn'

for seed in "${SEEDS[@]}"; do
    for norm in "${NORM[@]}"; do
        for mode in "${MODES[@]}"; do
            OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
                train_gpt2.py \
                --data_path "data/cn_wiki" \
                --gen_prompt "六四事件" \
                --device_batch_size 140 \
                --batch_size 140 \
                --num_iterations 5001 \
                --gen_every 500 \
                --save_every 500 \
                --train_loss_every 50 \
                --val_loss_every 50 \
                --n_heads 12 \
                --n_layers 12 \
                --head_dim 8 \
                --sequence_length 512 \
                --attn_mode "$mode" \
                --head_mode "euc"\
                --normalization "$norm"\
                --curvature 1. \
                --k_lr 1. \
                --seed "$seed" \
                --print_multiplier 5 \
                > run_logs/${NAME}_${norm}_${mode}_run_${seed}.txt 2>&1
        done
    done
done
