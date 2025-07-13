#!/bin/bash

NAME="shakespeare"
INIT_CURVS=(0.001 0.01 0.1 1.0 10.0)
SEEDS=(0 1 2 3)
MODES=('hyp')
LR=(1.)

for init_c in "${INIT_CURVS[@]}"; do
    for k_lr in "${LR[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for mode in "${MODES[@]}"; do
                OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
                    train_gpt2.py \
                    --data_path "data/shakespeare_char" \
                    --gen_prompt "THIBAULT: " \
                    --device_batch_size 64 \
                    --batch_size 64 \
                    --num_iterations 5001 \
                    --gen_every 500 \
                    --train_loss_every 50 \
                    --val_loss_every 50 \
                    --n_heads 8 \
                    --n_layers 8 \
                    --head_dim 8 \
                    --sequence_length 256 \
                    --attn_mode "$mode" \
                    --head_mode "euc"\
                    --curvature "$init_c" \
                    --k_lr "$k_lr" \
                    --seed "$seed" \
                    > logs/${NAME}_${mode}_c${init_c}_lr${k_lr}_${seed}.txt 2>&1
            done
        done
    done
done
