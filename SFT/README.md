# SFT

This folder contains the fine-tuning pipeline for different methods (Vanilla SFT, GIFT, CART, LIFT-2, LIFT-3).

## Run script

Use the launcher from repo root:

```bash
bash scripts/sft/run_sft.sh
```

Run LIFT with:

```bash
bash scripts/sft/run_lift_sft.sh
```

The default learning rate for LIFT is `5e-6`.

To run the two-component LIFT variant, change `--method lift3` to `--method lift2`.

Launcher command:

```bash
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
```

LIFT launcher command:

```bash
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
  --learning_rate 5e-6 \
  --train_data simplescaling/s1k \
  --method lift3 \
  --loss_selection time \
  --time_scaling
```

## Token frequency script

Use the token-frequency launcher from repo root:

```bash
bash scripts/sft/run_token_freq.sh
```

Launcher command:

```bash
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
```

## SFT datasets

`DatasetPreprocessor` in `sft_trainer.py` currently supports:

| `--train_data` |
| --- |
| `simplescaling/s1k` |
| `simplescaling/s1K-1.1` |
| `openai/gsm8k` |
| `divelab/dllm` |
| `divelab/LIFT-SFT-12K` |
| `KodCode/KodCode-Light-RL-10K` |

## Loss types and method mapping

Method-level interface (`--method`) in `sft_train.py`:

| `--method` | Default `--loss_type` |
| --- | --- |
| `vanilla_sft` | `vanilla` |
| `gift` | `GIFT` |
| `cart` | `CART` |
| `lift2` | `mixed` (2-component mix) |
| `lift3` | `mixed` (3-component mix) |

Implemented trainer loss modes (`dLLMTrainer.losses`):

| `--loss_type` | Behavior |
| --- | --- |
| `vanilla` | Standard masked-token SFT loss. |
| `GIFT` | Entropy-guided masking/reweighting style loss. |
| `CART` | Context-adaptive reweighting loss. |
| `mixed` | Mixture of token-selection losses (used by LIFT-2/LIFT-3). |
| `topk` | Top-k token confidence selection variant. |
| `bottomk` | Bottom-k token confidence selection variant. |

For `mixed`, `--loss_selection` controls policy (`time` or `random`).

## Merge LoRA checkpoint

After training, you can merge the LoRA checkpoint with the base model using the following command:

```bash
python SFT/merge_lora.py \
  --base_model GSAI-ML/LLaDA-8B-Instruct \
  --adapter_path SFT/sft_output/<run>/checkpoint-<step> \
  --output_path SFT/merged_models/<run>/checkpoint-<step> \
  --architecture llada
```

`--architecture llada` loads the base checkpoint with the repo's local LLaDA model class before merging, which is the path to use for evaluation with this repository. Use `--architecture auto` only when merging a non-LLaDA adapter that can be loaded directly with Hugging Face `AutoModel`.

If the adapter has `adapter_config.json`, the script uses its `base_model_name_or_path` value. `--base_model` is only needed as a fallback when that metadata is missing.
