#!/bin/bash
# SEEDS=(0)
# MODES=('hyp')

SEEDS=(0 1 2 3)
MODES=('hyp' 'euc')

NAME='ts'

{
  "test_run": false,
  "debug": false,
  "data_path": "data/cn_wiki",
  "batch_size": 64,
  "device_batch_size": 64,
  "num_iterations": 1001,
  "gen_every": 200,
  "gen_prompt": "\u5929\u5b89\u95e8\u5e7f\u573a",
  "train_loss_every": 20,
  "val_loss_every": 20,
  "head_dim": 16,
  "n_heads": 8,
  "n_layers": 8,
  "seed": 0,
  "sequence_length": 1024,
  "k_lr": 0.0,
  "curvature": 1.0,
  "head_mode": "euc",
  "attn_mode": "euc"
}

for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 \
            train_gpt2.py \
            --data_path "data/cn_wiki" \
            --gen_prompt "Lily " \
            --device_batch_size 64 \
            --batch_size 64 \
            --num_iterations 101 \
            --gen_every 500 \
            --train_loss_every 10 \
            --val_loss_every 10 \
            --n_heads 12 \
            --n_layers 12 \
            --head_dim 8 \
            --sequence_length 512 \
            --attn_mode "$mode" \
            --head_mode "euc"\
            --curvature 1. \
            --k_lr 1. \
            --seed "$seed" \
            > new_logs/${NAME}_${mode}_run_${seed}.txt 2>&1
    done
done
