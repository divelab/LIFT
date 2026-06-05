#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=LIFT_final \
accelerate launch \
  --config_file SFT/ddp_config.yaml \
  --main_process_port 29501 \
  --num_processes 4 \
  SFT/sft_train.py \
  --grad_accum_steps 2 \
  --batch_size 1 \
  --num_epochs 20 \
  --output_dir ./SFT/sft_output \
  --learning_rate 5e-6 \
  --train_data simplescaling/s1k \
  --method lift3 \
  --loss_selection time \
  --time_scaling
