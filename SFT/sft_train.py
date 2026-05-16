import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
from sft_trainer import *
import torch.distributed as dist
import random
import numpy as np
from datetime import datetime

def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Name of the pretrained model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--method",
        type=str,
        default="vanilla_sft",
        choices=["vanilla_sft", "gift", "cart", "lift2", "lift3"],
        help="Published fine-tuning method to run.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum sequence length for tokenization"
    )
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for the optimizer")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_output",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--job_name", type=str, default="llada-s1", help="Job Name")
    parser.add_argument("--train_data", type=str, default="simplescaling/s1k", help="Path to training data")
    parser.add_argument(
        "--debugging", action="store_true", help="Use while debugging model - only disables wandb logging"
    )
    parser.add_argument('--loss_type', type=str, default=None, help='Low-level trainer loss override.')
    parser.add_argument('--bottom_k_percent', type=float, default=None, help='Percentage of Hard Tokens to do SFT on')
    parser.add_argument('--loss_selection', type=str, default='time', choices=["random", "time"], help='Loss Selection For Mixed SFT (Time / Random)')
    parser.add_argument('--timestep_dist', type=str, default=None, help='Timestep distribution for diffusion sampling. Only discrete_uniform is supported.')  
    parser.add_argument('--fixed_timestep', type=float, default=False, help='Fixed timestep value (0.0 to 1.0). If set, uses this instead of random timesteps.')
    parser.add_argument('--time_scaling', action='store_true', help='Use Time Scaling in Loss Computation')
    parser.add_argument('--approximate', action='store_true', help='Approximate Time Based Masking')
    parser.add_argument('--complementary_mask', action='store_true', help='Enable Complementary Training')
    return parser.parse_args()


def resolve_method_args(args):
    method_to_loss = {
        "vanilla_sft": "vanilla",
        "gift": "GIFT",
        "cart": "CART",
        "lift2": "mixed",
        "lift3": "mixed",
    }
    if args.loss_type is None:
        args.loss_type = method_to_loss[args.method]
    args.lift_components = 2 if args.method == "lift2" else 3
    return args

# Model loading with LoRA integration
def load_model_and_tokenizer(args):
    from peft import LoraConfig, get_peft_model, TaskType

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    ) 

    # Load model
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Applying LoRA model
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)  # Cast fp32 lora params to bf16

    return tokenizer, model


# Dataset loading
def load_data(args, tokenizer):
    
    data = {}
    if args.train_data == "openai/gsm8k":
        data = load_dataset(args.train_data, "main", split="train")
    elif args.train_data in ["simplescaling/s1k", "simplescaling/s1K-1.1"]:
        data = load_dataset(args.train_data, split="train")
    elif args.train_data == "KodCode/KodCode-Light-RL-10K":
        data = load_dataset(args.train_data, split = "train")
    else:
        print(args.train_data)
        data = load_dataset("parquet",data_files=args.train_data, split='train')
    
    print(data)
    if 'math_combined' in args.train_data or 'math_set_combined' in args.train_data:
        args.train_data = 'divelab/dllm'
    
    preprocessor = DatasetPreprocessor(args.train_data)
    train_data, eval_data = preprocessor.preprocess_dataset(data, tokenizer, args.max_length)
        
    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
    
    return train_dataset, eval_dataset


# Training setup
def train_model(args, tokenizer, model):
    # Load dataset
    
    train_dataset, eval_dataset = load_data(args, tokenizer)

    # Training arguments setup
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f'{args.job_name}_{args.method}_loss_{args.loss_type}_selection_{args.loss_selection}'),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_steps=100,
        save_total_limit=20,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        lr_scheduler_type='linear',
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )

    # Create optimizer and scheduler
    num_train_steps = int(
        len(train_dataset)
        * args.num_epochs
        / (args.batch_size * args.grad_accum_steps * torch.cuda.device_count())
    )
    # Initialize Trainer with custom dLLMTrainer
    is_dream = "dream" in args.model_name.lower()
    dynamic_mask_token_id = 151666 if is_dream else 126336
    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(tokenizer=tokenizer, 
                                       mask_token_id=dynamic_mask_token_id, 
                                       max_length=args.max_length, 
                                       rdro_sampling=args.loss_type not in ['vanilla', 'focal', 'time_focal', 'CART', 'GIFT'],
                                       timestep_dist=args.timestep_dist, 
                                       fixed_timestep=args.fixed_timestep),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_type=args.loss_type,
        bottom_k_percent=args.bottom_k_percent,
        mix_policy=args.loss_selection,
        lift_components=args.lift_components,
        complementary_mask=args.complementary_mask,
        time_scaling=args.time_scaling,
        do_approximation=args.approximate,
        is_dream = is_dream
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    args = resolve_method_args(args)
    init_seed(args.seed)

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args)

    # Train the model
    train_model(args, tokenizer, model)
