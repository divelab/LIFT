import torch
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm


sys.path.append(str(Path(__file__).parent.parent))
from sft_trainer import dLLMSFTDataset, dLLMDataCollator, DatasetPreprocessor


def format_json_compact_lists(data):
    """Format JSON with lists on single lines."""
    lines = []
    lines.append('{')
    
    items = list(data.items())
    for i, (key, value) in enumerate(items):
        is_last = (i == len(items) - 1)
        lines.append(f'  "{key}": {{')
        
        # Format positions
        lines.append(f'    "positions": {json.dumps(value["positions"])},')
        # Format confidences
        lines.append(f'    "confidences": {json.dumps(value["confidences"])},')
        # Format masked_neighborhoods
        lines.append(f'    "masked_neighborhoods": {json.dumps(value["masked_neighborhoods"])},')
        # Format timesteps
        lines.append(f'    "timesteps": {json.dumps(value["timesteps"])}')
        
        if is_last:
            lines.append('  }')
        else:
            lines.append('  },')
    
    lines.append('}')
    return '\n'.join(lines)



def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Token Frequency Analysis")
    
    parser.add_argument(
        "--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Name of the pretrained model"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for data loading")
    
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum sequence length for tokenization"
    )
    parser.add_argument("--train_data", type=str, default="simplescaling/s1k", help="Path to training data")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./token_freq_logs",
        help="Directory to save json output",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to analyze (if not provided, processes entire dataset)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to iterate through the dataset"
    )
    parser.add_argument(
        "--fixed_timestep",
        type=float,
        default=None,
        help="Fixed timestep for masking (0.0 to 1.0). If None, uses discrete_uniform."
    )
    parser.add_argument(
        "--timestep_dist",
        type=str,
        default="discrete_uniform",
        help="Timestep distribution to use (only 'discrete_uniform' [1/16, 1/8, 1/4, 1/2, 1/sqrt(2), 1/sqrt(sqrt(2))] is currently supported)"
    )
    parser.add_argument(
        "--debugging", action="store_true", help="Debug mode"
    )
    return parser.parse_args()


def load_model_and_tokenizer(args):
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

    return tokenizer, model


from datasets import load_dataset

def load_data(args, tokenizer):
    if args.train_data == 'divelab/dllm':
        data = load_dataset(
            "parquet",
            data_files="/home/jotsna/math_dataset.parquet",
            split="train"
        )
    else:
        data = load_dataset(args.train_data, split="train")

    print(len(data))

    preprocessor = DatasetPreprocessor(args.train_data)
    train_data, eval_data = preprocessor.preprocess_dataset(
        data, tokenizer, args.max_length
    )

    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    print(f'Loaded {len(train_dataset)} training samples')

    return train_dataset



def analyze_token_frequencies(model, dataloader, device, tokenizer, output_file, num_samples=None, num_epochs=1):
    """
    Analyze ground truth tokens with their positions and confidences with dynamic logging.
    
    Returns:
        dict: {gt_token_id: {"positions": [...], "confidences": [...], "masked_neighborhoods": [...], "timesteps": [...]}}
    """
    model.eval()
    
    # Use defaultdict to automatically create lists for new tokens
    token_data = defaultdict(lambda: {"positions": [], "confidences": [], "masked_neighborhoods": [], "timesteps": []})
    
    total_masked = 0
    
    # Open file for dynamic logging
    file_handle = open(output_file, 'w')
    print(f"Logging to: {output_file}")
    
    try:
        with torch.no_grad():
            for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
                pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)
                for step, batch in pbar:
                    if num_samples is not None and step >= num_samples:
                        break
                    
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Get timestep from batch (added by collator)
                    timesteps = batch['t']  # Shape: (B, N)

                    # Get model predictions
                    outputs = model(input_ids)
                    logits = outputs.logits  # (B, N, V)

                    # Find masked positions (where labels != -100)
                    masked_positions = (labels != -100)

                    if not masked_positions.any():
                        continue
                    
                    # Compute probabilities
                    probs = F.softmax(logits, dim=-1)  # (B, N, V)

                    # Get argmax predictions
                    x0 = torch.argmax(logits, dim=-1)  # (B, N)

                    # Get confidence of predictions
                    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, N)

                    # Extract masked indices
                    masked_indices = masked_positions.nonzero(as_tuple=False)  # (num_masked, 2)

                    for b, n in masked_indices:
                        # Get ground truth token
                        gt_token = labels[b, n].item()

                        # Get confidence of predicted token (argmax)
                        confidence = x0_p[b, n].item()

                        # Get position
                        position = n.item()
                        
                        # Get timestep for this position
                        timestep = timesteps[b, n].item()
                        
                        # Compute masked neighborhood (count surrounding masked positions)
                        seq_len = masked_positions.shape[1]
                        left_count = 0
                        right_count = 0
                        
                        # Count masked positions to the left
                        for i in range(position - 1, -1, -1):
                            if masked_positions[b, i]:
                                left_count += 1
                            else:
                                break
                        
                        # Count masked positions to the right
                        for i in range(position + 1, seq_len):
                            if masked_positions[b, i]:
                                right_count += 1
                            else:
                                break
                        
                        masked_neighborhood = left_count + right_count

                        # Store in dictionary
                        token_data[gt_token]["positions"].append(position)
                        token_data[gt_token]["confidences"].append(confidence)
                        token_data[gt_token]["masked_neighborhoods"].append(masked_neighborhood)
                        token_data[gt_token]["timesteps"].append(timestep)

                        total_masked += 1
                    
                    # Update progress bar
                    pbar.set_postfix({"masked_tokens": total_masked, "unique_tokens": len(token_data)})

                    if (step + 1) % 10 == 0:
                        file_handle.seek(0)
                        file_handle.truncate()
                        # Format data with just token IDs
                        formatted_data = {}
                        for token_id, data in token_data.items():
                            formatted_data[str(token_id)] = {
                                "positions": data["positions"],
                                "confidences": data["confidences"],
                                "masked_neighborhoods": data["masked_neighborhoods"],
                                "timesteps": data["timesteps"]
                            }
                        file_handle.write(format_json_compact_lists(formatted_data))
                        file_handle.flush()
    finally:
        # Final write
        file_handle.seek(0)
        file_handle.truncate()
        # Format data with just token IDs
        formatted_data = {}
        for token_id, data in token_data.items():
            formatted_data[str(token_id)] = {
                "positions": data["positions"],
                "confidences": data["confidences"],
                "masked_neighborhoods": data["masked_neighborhoods"],
                "timesteps": data["timesteps"]
            }
        file_handle.write(format_json_compact_lists(formatted_data))
        file_handle.close()
        print(f"Closed log file")
    
    # Convert defaultdict to regular dict for JSON serialization
    token_data = dict(token_data)
    
    print(f"\nTotal masked tokens: {total_masked}")
    print(f"Unique ground truth tokens: {len(token_data)}")
    
    return token_data

def main():
    init_seed(42)
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(args)
    
    # Load data
    print("Loading data...")
    train_dataset = load_data(args, tokenizer)
    
    collator_kwargs = {
        "tokenizer": tokenizer,
        "mask_token_id": 126336,
        "max_length": args.max_length,
        "timestep_dist": args.timestep_dist,
    }
    
    if args.fixed_timestep is not None:
        collator_kwargs["fixed_timestep"] = args.fixed_timestep
        print(f"Using fixed timestep: {args.fixed_timestep}")
    else:
        collator_kwargs["fixed_timestep"] = False
        print(f"Using timestep distribution: {args.timestep_dist}")
        print(f"Discrete timesteps: [1/16, 1/8, 1/4, 1/2, 1/√2, 1/2^0.25]")
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=dLLMDataCollator(**collator_kwargs),
        shuffle=False
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Prepare output file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"token_freq_{timestamp}.json"
    
    # Analyze token frequencies with dynamic logging
    if args.num_samples is None:
        print(f"\nAnalyzing token frequencies for entire dataset over {args.num_epochs} epochs...")
    else:
        print(f"\nAnalyzing token frequencies for {args.num_samples} samples per epoch over {args.num_epochs} epochs...")
    
    token_data = analyze_token_frequencies(
        model, 
        dataloader, 
        device,
        tokenizer,
        output_file=str(output_file),
        num_samples=args.num_samples,
        num_epochs=args.num_epochs
    )
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Unique tokens: {len(token_data)}")
    
    # Get top 10 most frequent tokens
    token_counts = {token: len(data["positions"]) for token, data in token_data.items()}
    top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 most frequent masked tokens:")
    for token_id, count in top_tokens:
        token_str = tokenizer.decode([int(token_id)])
        avg_conf = np.mean(token_data[token_id]["confidences"])
        print(f"  Token {token_id} ('{token_str}'): {count} occurrences, avg confidence: {avg_conf:.4f}")
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()