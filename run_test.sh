#!/bin/bash

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone --nproc_per_node=1 \
    train_gpt2_main.py \
    --test_run > logs/test.txt 2>&1