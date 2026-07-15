<div align="center">
  <h1>Learnability-Informed Fine-Tuning for Diffusion Language Models</h1>
  <p>Code for supervised fine tuning and evaluating diffusion LLMs with Learnability-Informed Fine-Tuning (LIFT).</p>
  <p>
    <a href="https://arxiv.org/abs/2605.22939"><strong>Paper</strong></a> &nbsp;|&nbsp;
    <a href="https://huggingface.co/papers/2605.22939"><strong>Hugging Face</strong></a> 
  </p>
</div>

<div align="center">
  <img src="media/LIFT_framework.png" alt="Learnability-Informed Fine-Tuning results" width="100%" />

</div>


<div align="center">
  <hr width="100%">
</div>

## Updates

- **April 2026**: Accepted to ICML 2026!
- **July 2026**: Released `LIFT-SFT-12K` on Hugging Face: [divelab/LIFT-SFT-12K](https://huggingface.co/datasets/divelab/LIFT-SFT-12K)

## Repository Layout

```text
scripts/   launch scripts
SFT/       fine-tuning and LoRA merge
eval/      evaluation, generation, and scoring
dataset/   local datasets (countdown/sudoku/AIME JSONs)
```

## Environment Setup

```bash
conda env create -f lift.yml
conda activate lift
```

## SFT

Run SFT with the root launcher:

```bash
bash scripts/sft/run_sft.sh
```

For LIFT training, use the dedicated launcher with the default learning rate set to `5e-6`:

```bash
bash scripts/sft/run_lift_sft.sh
```

Merge LoRA adapters for standalone evaluation checkpoints:

```bash
python SFT/merge_lora.py \
  --base_model GSAI-ML/LLaDA-8B-Instruct \
  --adapter_path SFT/sft_output/<run>/checkpoint-<step> \
  --output_path SFT/merged_models/<run>/checkpoint-<step> \
  --architecture llada
```

Use `--architecture llada` for checkpoints that will be evaluated with this repo's LLaDA evaluation code. The script reads `base_model_name_or_path` from the adapter config when available; `--base_model` is used as the fallback.

## Evaluation

Run evaluation with:

```bash
bash scripts/eval/run_eval.sh
```

The eval runner prints accuracy when generation finishes and writes the score into each generation JSON.

Supported `--dataset` keys in `eval/eval.py`:

`gsm8k`, `math500`, `countdown`, `sudoku`, `aime24`, `aime25`, `humaneval`, `mbpp`



## Module Docs

- `SFT/README.md` for training methods, datasets, and SFT scripts
- `eval/README.md` for evaluation workflow and task details
