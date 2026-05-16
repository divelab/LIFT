import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
from generate import generate
import random
import re
import json
from pathlib import Path
from gsm8k import GSM8KDataset
from datasets import load_dataset
from parsers import Parser, is_equiv

MATH_DATASET_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>" 
"""

class MathDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        json_path=str(Path(__file__).resolve().parents[1] / "dataset" / "aime25.json"),
        num_examples=0,
        add_reasoning=True,
        system_prompt=MATH_DATASET_SYSTEM_PROMPT,
        subsample=-1,
    ):
        self.json_path = json_path
        super().__init__(
            tokenizer=tokenizer,
            num_examples=num_examples,
            add_reasoning=add_reasoning,
            system_prompt=system_prompt,
            subsample=subsample,
        )

    # -------------------------------
    # Load test dataset
    # -------------------------------
    def load_dataset(self):
        with open(self.json_path, "r") as f:
            self.dataset = json.load(f)

    # -------------------------------
    # Few-shot examples
    # -------------------------------
    def load_few_shot_examples(self):
        if self.num_examples <= 0:
            return []

        indices = random.sample(range(len(self.dataset)), self.num_examples)
        few_shot_examples = []

        for idx in indices:
            row = self.dataset[idx]
            few_shot_examples.append({
                "question": row["question"],
                "answer": row["sol"][0] if row.get("sol") else ""
            })

        return few_shot_examples

    # -------------------------------
    # Get item
    # -------------------------------
    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]

        question = row["question"]
        answer = row["sol"][0] if row.get("sol") else ""
        prompt = self.create_prompt(question)

        return prompt, question, answer

    def __len__(self):
        return len(self.subsample)
