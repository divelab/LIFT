#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  --master_port 29527 \
  eval/eval.py \
  --dataset gsm8k \
  --batch_size 32 \
  --gen_length 256 \
  --block_length 32 \
  --diffusion_steps 128 \
  --output_dir results \
  --model_path SFT/merged_models/llada-s1_lift3_loss_mixed_selection_time/checkpoint-124
