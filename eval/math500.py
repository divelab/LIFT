import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
import random
import re
from gsm8k import GSM8KDataset
from datasets import load_dataset, concatenate_datasets
from parsers import Parser
MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>" 
"""


class MATH500Dataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        split='test',
        add_reasoning=True,
        system_prompt=MATH500_SYSTEM_PROMPT,
        subsample=-1,
    ):  
        self.subsets = ["algebra", "counting_and_probability", "geometry","intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
        super().__init__(tokenizer, num_examples, add_reasoning, split, system_prompt, subsample)

    def load_dataset(self, split='test'):
        loaders = {
            "train": lambda: concatenate_datasets([
                    load_dataset("EleutherAI/hendrycks_math", subset, split="train")
                    for subset in self.subsets
            ]).shuffle(seed=42).map(
                lambda x: {"problem": x["problem"], "answer": Parser.extract_answer_boxed(x["solution"])}
                # remove_columns=["problem", "solution"]
            ),
            "test": lambda: load_dataset("HuggingFaceH4/MATH-500", split="test"),
        }

        self.dataset = loaders[split]()
        print(f"Loaded MATH {split} dataset with {len(self.dataset)} examples")

    def load_few_shot_examples(self):
        train_data = load_dataset("EleutherAI/hendrycks_math", ("algebra"), split="train")
        few_shot_examples = []
        samples = random.sample(range(len(train_data)), self.num_examples)
        for example in samples:
            few_shot_examples.append(
                {"question": train_data[example]["problem"], "answer": train_data[example]["solution"]}
            )
        return few_shot_examples

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["problem"]
        answer = self.dataset[self.subsample[idx].item()]["answer"]
        prompt = self.create_prompt(question)
        return {
            "prompt": prompt, 
            "question": question, 
            "answer": answer
        }
