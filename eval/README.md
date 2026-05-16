# eval

This folder contains the evaluation pipeline for LIFT.

## Main entrypoints

- `eval.py`: distributed generation/evaluation runner
- `scoring.py`: task-specific parsing and accuracy scoring used by `eval.py`

## Quick run (from repo root)

```bash
bash scripts/eval/run_eval.sh
```

Accuracy is printed by the eval runner when generation finishes.

## Supported datasets

`eval.py --dataset` supports:

`gsm8k`, `math500`, `countdown`, `sudoku`, `aime24`, `aime25`, `humaneval`, `mbpp`


## File guide

| File | Role |
| --- | --- |
| `eval.py` | Loads model/checkpoint, builds task dataset, runs distributed generation, prints accuracy, writes generation JSON. |
| `generate.py` | Diffusion decoding routines (including cache-enabled variants). |
| `scoring.py` | End-to-end parser/scorer for all supported tasks. |
| `parsers.py` / `parser_helper.py` / `auto_scoring_judge.py` | Task-specific parsing and correctness checks. |
| `gsm8k.py`, `math500.py`, `math_dataset.py`, `countdown.py`, `sudoku.py`, `humaneval.py`, `mbpp.py` | Dataset adapters and prompt formatting. |
| `model/` | Local LLaDA model config/implementation used for non-Dream loading. |
