#!/usr/bin/env python3
"""
Script to merge LoRA adapter weights with base model and save as a full model
for inference evaluation.

python merge_lora.py \
    --base_model GSAI-ML/LLaDA-8B-Instruct \
    --adapter_path ./sft_output/llada-s1_loss_vanilla_selection_random/checkpoint-7425 \
    --output_path ./merged_models/llada-s1_loss_vanilla_selection_random/checkpoint-7425
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from model.modeling_llada import LLaDAModelLM


def infer_base_model_path(adapter_path, fallback):
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return fallback

    with adapter_config_path.open() as f:
        adapter_config = json.load(f)
    return adapter_config.get("base_model_name_or_path") or fallback


def load_base_model(base_model_path, architecture):
    if architecture == "llada":
        return LLaDAModelLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
        )

    return AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )



def merge_lora_model(base_model_path, adapter_path, output_path, architecture):
    """
    Merge LoRA adapter with base model and save as full model
    """
    base_model_path = infer_base_model_path(adapter_path, base_model_path)
    print(f"Loading base model from: {base_model_path}")

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=True
    )

    base_model = load_base_model(base_model_path, architecture)

    print(f"Loading LoRA adapter from: {adapter_path}")

    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA weights with base model...")

    # Merge and unload
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")

    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("✅ Model merged and saved successfully!")
    print(f"You can now use this path for inference: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Base model path")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for merged model")
    parser.add_argument(
        "--architecture",
        type=str,
        default="llada",
        choices=["llada", "auto"],
        help="Model loader to use for the base checkpoint. Use llada for repo-native LLaDA eval.",
    )

    args = parser.parse_args()

    merge_lora_model(args.base_model, args.adapter_path, args.output_path, args.architecture)
