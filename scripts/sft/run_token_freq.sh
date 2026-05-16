#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=0 \
python SFT/token_freq.py \
  --model_name GSAI-ML/LLaDA-8B-Instruct \
  --train_data simplescaling/s1k \
  --batch_size 1 \
  --max_length 4096 \
  --num_samples 20 \
  --num_epochs 1 \
  --timestep_dist discrete_uniform \
  --output_dir ./SFT/token_freq_logs
