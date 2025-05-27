#!/bin/bash
SEEDS=(0)
MODES=('euc' 'hyp')
NAME='cn'
# SEEDS=(0 1 2 3 5)
for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 \
            train_gpt2.py \
            --data_path "data/cn_wiki" \
            --gen_prompt "天安门广场" \
            --device_batch_size 64 \
            --batch_size 64 \
            --num_iterations 1001 \
            --gen_every 200 \
            --train_loss_every 20 \
            --val_loss_every 20 \
            --n_heads 8 \
            --n_layers 8 \
            --head_dim 16 \
            --sequence_length 1024 \
            --attn_mode "$mode" \
            --head_mode "euc"\
            --curvature 1. \
            --k_lr 0. \
            --seed "$seed" \
            > new_logs/${NAME}_${mode}_run_${seed}.txt 2>&1
    done
done
