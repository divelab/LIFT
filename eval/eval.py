import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from generate import generate
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset
from math_dataset import MathDataset
from humaneval import HumanEvalDataset
from mbpp import MBPPDataset
from scoring import score_generations

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "math500": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "aime24": MathDataset,
    "aime25": MathDataset,
    "humaneval": HumanEvalDataset,
    "mbpp": MBPPDataset,
}

AIME_JSON = {
    "aime24": Path(__file__).resolve().parents[1] / "dataset" / "aime24.json",
    "aime25": Path(__file__).resolve().parents[1] / "dataset" / "aime25.json",
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    is_dream = False,
    mask_id=126336,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        out = generate(
            model,
            input_ids,
            tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
            is_dream=is_dream,
            mask_id=mask_id,
        )

        generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(gt_answers))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


def aggregate_score_metrics(local_score, device, gen_length):
    score_tensor = torch.tensor(
        [
            float(local_score["correct"]),
            float(local_score["processed"]),
            float(local_score["total_effective_tokens"]),
            float(local_score["total_steps_decoded"]),
            float(local_score["steps_observed"]),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(score_tensor, op=dist.ReduceOp.SUM)

    correct = int(score_tensor[0].item())
    processed = int(score_tensor[1].item())
    total_effective_tokens = float(score_tensor[2].item())
    total_steps_decoded = float(score_tensor[3].item())
    steps_observed = int(score_tensor[4].item())

    accuracy = (correct / processed * 100) if processed > 0 else 0.0
    avg_effective_tokens = (total_effective_tokens / processed) if processed > 0 else 0.0
    avg_steps_decoded = (total_steps_decoded / steps_observed) if steps_observed > 0 else None
    avg_nfe = (gen_length / avg_steps_decoded) if avg_steps_decoded else None

    return {
        "correct": correct,
        "processed": processed,
        "accuracy": accuracy,
        "total_effective_tokens": total_effective_tokens,
        "avg_effective_tokens": avg_effective_tokens,
        "total_steps_decoded": total_steps_decoded,
        "steps_observed": steps_observed,
        "avg_steps_decoded": avg_steps_decoded,
        "avg_nfe": avg_nfe,
    }


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    init_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "math500", "countdown", "sudoku", "aime24", "aime25", "humaneval", "mbpp"],
        default="gsm8k",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json_path", type=str, default=None, help="Override JSON path for AIME-style datasets.")
    parser.add_argument(
        "--code_eval_output_dir",
        type=str,
        default="tmp_exec/",
        help="Temporary execution/output directory for HumanEval and MBPP scoring.",
    )
    args = parser.parse_args()

    local_rank = setup_ddp()
    print(f'Getting Results for: {args.dataset}')
    # args.diffusion_steps = args.gen_length // 2
    num_evals = {
        "gsm8k": -1,
        "math": -1,
        "math500": -1,
        "countdown": 256,
        "sudoku": 256,
        "aime24": -1,
        "aime25": -1,
        "humaneval": -1,
        "mbpp": -1,
    }
    
    is_dream = "dream" in args.model_path.lower()
    dynamic_mask_id = 151666 if is_dream else 126336

    if is_dream:
        print("Loading Dream model architecture...")
        model = AutoModel.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).to(local_rank)
        tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    else:
        print("Loading LLaDA model architecture...")
        model = LLaDAModelLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16
        ).to(local_rank)
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    # --------------------------------
    

    if args.checkpoint_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(
            local_rank
        )

        if dist.get_world_size() > 1:
            dist.barrier()  # Make sure all processes are ready
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")

    dataset_kwargs = {
        "tokenizer": tokenizer,
        "subsample": num_evals[args.dataset],
        "num_examples": args.few_shot,
        "add_reasoning": True,
    }
    if args.dataset in AIME_JSON:
        dataset_kwargs["json_path"] = args.json_path or str(AIME_JSON[args.dataset])
    if args.dataset in {"humaneval", "mbpp"}:
        dataset_kwargs["output_dir"] = args.code_eval_output_dir
    dataset = DATASET_MAP[args.dataset](**dataset_kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        #model_name = "instruct" if "Instruct" in args.model_path else "base"
        model_name = args.model_path.replace("/", "-").replace(".", "-")

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    
    time_str = time.strftime("%Y%m%d-%H%M%S")
    # cleansed_model_name = model_name.replace("/mnt/data/shared/shparashar/llada-play/SFT/sft_output/", "")
    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}_{dist.get_rank()}_{time_str}_generations.json"
    print(f"Saving generations to {filename}")

    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        temperature=args.temperature,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        is_dream=is_dream,       # <-- PASS TO EVALUATE
        mask_id=dynamic_mask_id,
    )

    local_score = score_generations(
        args.dataset,
        metrics["generations"],
        output_dir=args.code_eval_output_dir,
    )
    global_score = aggregate_score_metrics(local_score, model.device, args.gen_length)

    if dist.get_rank() == 0:
        avg_nfe = global_score["avg_nfe"]
        avg_nfe_str = f"{avg_nfe:.4f}" if avg_nfe is not None else "N/A"
        print("=" * 80)
        print(
            f"Final {args.dataset} accuracy: "
            f"{global_score['correct']}/{global_score['processed']} "
            f"({global_score['accuracy']:.2f}%)"
        )
        print(f"Avg effective tokens: {global_score['avg_effective_tokens']:.2f}")
        print(f"Avg NFE: {avg_nfe_str}")
        print("=" * 80)

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                        "local_score": {
                            key: value
                            for key, value in local_score.items()
                            if key != "detailed_results"
                        },
                        "global_score": global_score,
                    },
                    "parsed_results": local_score["detailed_results"],
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                },
                f,
                indent=2,
            )

    cleanup_ddp()
