#!/bin/bash

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,3 torchrun --standalone --nproc_per_node=2 \
    train_gpt2_main.py \
    --data_path "data/fineweb10B" \
    --device_batch_size 32 \
    --batch_size 512 \
    --num_iterations 4578 \
    --generate_every 0 \
    --n_heads 6 \
    --n_layers 12 \
    --head_dim 128 \
    --sequence_length 1024 \
    --attn_mode "euc" \
    --head_mode "euc" \
    --curvature 1.0 \
    --k_lr 0.0 \
    --seed 0 \
    > logs/last_fineweb.txt 2>&1