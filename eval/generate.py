import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)

# def get_num_transfer_tokens_dual_cache(block_mask_index, steps_per_block):
#     """
#     block_mask_index: (B, block_len) bool
#     returns: (B, steps_per_block) long
#     """
#     B = block_mask_index.shape[0]
#     device = block_mask_index.device

#     # Total masked tokens per batch
#     total = block_mask_index.sum(dim=1)  # (B,)

#     base = total // steps_per_block
#     remainder = total % steps_per_block

#     # Create step indices [0, 1, ..., steps_per_block-1]
#     step_ids = torch.arange(steps_per_block, device=device).unsqueeze(0)  # (1, S)

#     # Distribute remainder without Python branching
#     extra = (step_ids < remainder.unsqueeze(1)).long()  # (B, S)

#     return base.unsqueeze(1) + extra



@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    is_dream=False,
):
    """
    Optimized version of the generate function.
    """
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
                
                if is_dream:
                    dummy = torch.zeros_like(logits[:, :1, :])
                    logits = torch.cat([dummy, logits[:, :-1, :]], dim=1)

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)            # Ensure we don't process tokens beyond the current block
               
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
        return x

@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Prefix-cached version of the FIRST (optimized) generate().
    """

    device = prompt.device

    with torch.autocast(device_type="cuda"):
        # Initialize sequence
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()
        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        for num_block in range(num_blocks):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = block_start + block_length

            block_mask_index = (x[:, block_start:block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            # ---- Initial full forward pass (build cache) ----
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                output = model(x_, use_cache=True)
                logits, un_logits = torch.chunk(output.logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                past_key_values = output.past_key_values[:len(output.past_key_values)//2]
            else:
                output = model(x, use_cache=True)
                logits = output.logits
                past_key_values = output.past_key_values

            # ---- First update ----
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = False

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float)
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            for b in range(x.shape[0]):
                k = num_transfer_tokens[b, 0].item()
                if k > 0:
                    _, idx = torch.topk(confidence[b], k)
                    x[b, idx] = x0[b, idx]

            # ---- Trim cache to prefix only ----
            trimmed_past = []
            for layer in past_key_values:
                trimmed_layer = tuple(
                    kv[:, :, :block_start] for kv in layer
                )
                trimmed_past.append(trimmed_layer)
            past_key_values = trimmed_past

            # ---- Iterative refinement using prefix cache ----
            for i in range(1, steps_per_block):
                if (x[:, block_start:block_end] == mask_id).sum() == 0:
                    break

                mask_index = (x[:, block_start:] == mask_id)
                mask_index[:, block_length:] = False

                logits = model(
                    x[:, block_start:],
                    past_key_values=past_key_values,
                    use_cache=True,
                ).logits

                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                else:
                    x0_p = torch.rand_like(x0, dtype=torch.float)

                confidence = torch.where(mask_index, x0_p, -torch.inf)

                for b in range(x.shape[0]):
                    k = num_transfer_tokens[b, i].item()
                    if k > 0:
                        _, idx = torch.topk(confidence[b], k)
                        x[b, block_start:][idx] = x0[b, idx]

        return x

@torch.no_grad()
@torch.compile(mode="max-autotune", fullgraph=True)
def generate_with_dual_cache(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    B, Lp = prompt.shape
    device = prompt.device

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    # Initialize sequence
    x = torch.full(
        (B, Lp + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :Lp] = prompt

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # How many tokens to reveal per step
        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # ---- 1) Warm full cache once per block ----
        if cfg_scale > 0.0:
            prompt_index = (x != mask_id)
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)

            out = model(x_, use_cache=True)
            logits, un_logits = torch.chunk(out.logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            past_key_values = out.past_key_values[: len(out.past_key_values) // 2]
        else:
            out = model(x, use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values

        # ---- Trim cache to prefix only ----
        trimmed_past = []
        for layer in past_key_values:
            trimmed_past.append(
                tuple(kv[:, :, :s] for kv in layer)
            )
        past_key_values = tuple(trimmed_past)

        # ---- Fixed refinement loop (compile-friendly) ----
        for i in range(steps_per_block):
            # Predict only the current block using prefix cache
            out_blk = model(
                x[:, s:e],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits_blk = out_blk.logits

            # Sampling
            logits_noise = add_gumbel_noise(logits_blk, temperature)
            x0_blk = torch.argmax(logits_noise, dim=-1)

            # Confidence
            if remasking == "low_confidence":
                p = F.softmax(logits_blk, dim=-1)
                conf = torch.gather(p, -1, x0_blk.unsqueeze(-1)).squeeze(-1)
            else:
                conf = torch.rand_like(x0_blk, dtype=torch.float)

            # Only masked positions
            mask_blk = (x[:, s:e] == mask_id)
            conf = torch.where(mask_blk, conf, -torch.inf)

            # Top-k per batch
            new_blk = x[:, s:e]
            for b in range(B):
                k = num_transfer_tokens[b, i].item()
                if k > 0:
                    _, idx = torch.topk(conf[b], k)
                    new_blk[b, idx] = x0_blk[b, idx]

            # Merge block back (static concat)
            x = torch.cat([x[:, :s], new_blk, x[:, e:]], dim=1)

    return x
