#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
  --config_file SFT/ddp_config.yaml \
  --main_process_port 29501 \
  --num_processes 2 \
  SFT/sft_train.py \
  --grad_accum_steps 2 \
  --batch_size 2 \
  --num_epochs 5 \
  --output_dir ./SFT/sft_output \
  --learning_rate 1e-5 \
  --train_data simplescaling/s1k \
  --method vanilla_sft \
  --time_scaling
