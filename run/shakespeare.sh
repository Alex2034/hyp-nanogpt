#!/bin/bash

NAME="shakespeare"

SEEDS=(0 1 2 3 4)
MODES=('hyp')
NORM=('learnable' 'power' 'exp')

for norm in "${NORM[@]}"; do
    for seed in "${SEEDS[@]}"; do
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
            train_gpt2.py \
            --data_path "data/shakespeare_char" \
            --gen_prompt "THIBAULT:" \
            --device_batch_size 64 \
            --batch_size 64 \
            --num_iterations 4001 \
            --gen_every 500 \
            --train_loss_every 50 \
            --val_loss_every 50 \
            --n_layers 8 \
            --n_heads 4 \
            --head_dim 16 \
            --sequence_length 256 \
            --attn_mode "hyp" \
            --head_mode "euc" \
            --normalization "$norm" \
            --curvature 1. \
            --k_lr 1. \
            --seed "$seed" \
        > run_logs/shakespeare_${norm}_${seed}.txt 2>&1  # ${NAME}_${mode}_c${init_c}_lr${k_lr}_${seed}
    done
done
