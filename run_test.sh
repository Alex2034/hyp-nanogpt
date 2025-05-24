#!/bin/bash

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone --nproc_per_node=1 \
    train_gpt2_min.py \
    --test_run --debug > logs/test.txt 2>&1
