#!/bin/bash

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5 torchrun \
    --standalone --nproc_per_node=1 \
    train_gpt2.py \
    --num_iterations 200 \
    --train_loss_every 10 \
    --val_loss_every 10 \
    --attn_mode 'hyp' \
    --k_lr 1.0 \
    > logs/test.txt 2>&1
