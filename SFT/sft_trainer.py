import torch
import torch.nn.functional as F
import math
import numpy as np
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
from functools import partial

class dLLMTrainer(Trainer):
    def __init__(self, loss_type='vanilla', bottom_k_percent = None, seed=1234, mask_token_id=126336, mix_vanilla_coef=3, mix_policy ='random', lift_components=3, complementary_mask = False, time_scaling=True, do_approximation=False, is_dream=False, cart_p=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_dream = is_dream
        self.bottom_k_percent = bottom_k_percent  # Will be set if using bottom-r training
        self.rng = random.Random(seed)
        self.loss_mode = loss_type
        self.losses = {
            'vanilla': self.vanilla_sft_loss_old,
            'bottomk': self.bottom_k,
            'topk': self.top_k,
            'mixed': self.mixture_sft_loss,
            'CART': self.cart_loss,
            'GIFT': self.gift_loss,
        }
        self.mask_token_id = mask_token_id
        self.lift_components = lift_components
        self.mix_modes = self._build_mix_modes(lift_components)
        self.mix_probs = [1 / len(self.mix_modes)] * len(self.mix_modes)
        self._mix_cached_gs = None
        self._mix_cached_idx = None
        self.mix_vanilla_coef = mix_vanilla_coef
        self.mix_policy = mix_policy
        self.mix_counts = {"vanilla": 0, "topk": 0, "bottomk": 0}
        self.mix_total = 0
        self.complementary_training = complementary_mask
        self.use_time_scaling = time_scaling
        self.approximation = do_approximation
        self.cart_p = cart_p
        # Focal Loss
        print(f'Using Time Scaling? {self.use_time_scaling} - Approximation ? - {self.approximation}')
        self.focal_gamma = 2
    
    # Need to Support Per Device Batching Later

    def _build_mix_modes(self, lift_components):
        modes = [
            ("topk", partial(self.bottom_k, top_k=True)),
            ("bottomk", partial(self.bottom_k, top_k=False)),
        ]
        if lift_components == 3:
            return [("vanilla", self.vanilla_mix_loss)] + modes
        if lift_components == 2:
            return modes
        raise ValueError("lift_components must be 2 or 3")
    
    def _shift_logits_if_dream(self, logits):
        if self.is_dream:
            dummy = torch.zeros_like(logits[:, :1, :])
            return torch.cat([dummy, logits[:, :-1, :]], dim=1)
        return logits

    def get_logits(self, model, model_inputs):
        return self._shift_logits_if_dream(model(**model_inputs).logits)
    
    def _mix_probs_from_t(self, t):
        t_b = t[:, 0] if t.ndim > 1 else t
        t_mean = float(t_b.mean().item())
        if self.lift_components == 2:
            s = max(t_mean + (1.0 - t_mean), 1e-8)
            return [t_mean / s, (1.0 - t_mean) / s]

        w_top = t_mean
        w_bottom = 1.0 - t_mean
        w_van = self.mix_vanilla_coef * t_mean * (1.0 - t_mean)
        s = w_top + w_bottom + w_van
        probs = [w_van / s, w_top / s, w_bottom / s]
        return probs
    
    def _get_mix_probs(self, t=None):
        policies = {
            "random":  lambda _t: self.mix_probs,
            "time":    lambda _t: self._mix_probs_from_t(_t),
        }
        return policies[self.mix_policy](t)
          
    def build_complimentary_views(self, input_ids, selected_mask, selected_mask_labels, clean_input, prompt_mask, t):
        
        device = input_ids.device
        B, L = input_ids.shape
        
        # Complimentary Mask
        complimentary_mask = ~prompt_mask & ~selected_mask
        
        # Applying Complimentary Mask 
        complimentary_input_ids = clean_input.clone()
        complimentary_input_ids[complimentary_mask] = self.mask_token_id
        
        # Generating Labels
        complimentary_labels = clean_input.clone()
        complimentary_labels[~complimentary_mask] = -100
        
        t_b = t[:, 0].float() if t.ndim == 2 else t.float()
        tcompliment_b = (1.0 - t_b).clamp(min=1e-3)
        tcompliment_b = tcompliment_b[:, None].expand(B,L)
        
        # Concatenating Inputs, Labels and Time
        # print(f'Sums Here - {complimentary_mask.sum()} - {selected_mask.sum()} - {prompt_mask.sum()}')
        input_ids_cat = torch.cat([input_ids, complimentary_input_ids], dim=0).to(device)
        labels_cat = torch.cat([selected_mask_labels, complimentary_labels], dim=0).to(device)
        t_cat = torch.cat([t, tcompliment_b], dim=0).to(device)
        return input_ids_cat, labels_cat, t_cat
         
      
    def _get_mix_idx(self, t=None):
        global_step = int(self.state.global_step)
        # Ensure the same loss was used in the grad-accum steps.
        # if self._mix_cached_gs == global_step:
        #     return self._mix_cached_idx

        mix_probs = self._get_mix_probs(t)
        idx = int(self._sample_idx(mix_probs))
        self._mix_cached_gs = global_step
        self._mix_cached_idx = idx
        return idx
    
    def _sample_idx(self, probs):
        sampling_policy = {
            'random': self.rng.choices(range(len(self.mix_modes)), weights=probs, k=1)[0],
            'time': int(torch.tensor(probs).argmax().item())
        }
        return sampling_policy[self.mix_policy]
        
    # Mix B/W Vanilla, Bottom K and TopK
    def mixture_sft_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        t = inputs.get("t", None)
        idx = self._get_mix_idx(t=t)
        loss_name, loss_fn = self.mix_modes[idx]
        self.mix_counts[loss_name] += 1
        self.mix_total += 1
        if (self.state.global_step) % self.args.logging_steps == 0:
            self.log({
                "frac_vanilla": float(self.mix_counts["vanilla"] / max(self.mix_total, 1)),
                "frac_topK": float(self.mix_counts["topk"] / max(self.mix_total, 1)),
                "frac_bottomK": float(self.mix_counts["bottomk"] / max(self.mix_total, 1)),
            })
        
        loss, outputs = loss_fn(model, inputs, num_items_in_batch, return_outputs=True)
        print(f'RANK [{dist.get_rank()}] Loss Name: {loss_name} - Global Step: {self._mix_cached_gs} - Mini Batch Loss: {loss} - Time: {t.mean()}\n')
        return (loss, outputs) if return_outputs else loss

    def _context_adaptive_reweight(self, seq_len, device):
        position_ids_l = np.arange(seq_len).reshape(-1, 1)
        position_ids_r = np.arange(seq_len).reshape(1, -1)
        distance = torch.from_numpy(position_ids_l - position_ids_r)
        if not 0 < self.cart_p < 1:
            raise ValueError("cart_p must be in (0, 1)")
        matrix = (math.log(self.cart_p) + (distance.abs() - 1) * math.log(1 - self.cart_p)).exp() * 0.5
        matrix.masked_fill_(distance == 0, 0)
        return matrix.to(device)

    def cart_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        inputs.pop("rho")
        inputs.pop("prompt_mask")
        inputs.pop("clean_inputs")
        inputs.pop("pad_token_id", None)
        inputs.pop("mask_token_id", None)

        logits = self.get_logits(model, inputs)
        outputs = model(**inputs) if return_outputs else None
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none", ignore_index=-100
        ).view(logits.shape[0], -1)

        masked_positions = labels != -100
        if not masked_positions.any():
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return loss if not return_outputs else (loss, outputs)

        weight_matrix = self._context_adaptive_reweight(labels.shape[-1], labels.device)
        non_mask = ~masked_positions
        weights = non_mask.type_as(weight_matrix).matmul(weight_matrix).masked_fill(non_mask, 0)
        unscaled_loss = unscaled_loss * weights

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / masked_positions.sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss if not return_outputs else (loss, outputs)

    def gift_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")
        t = inputs.pop("t")
        inputs.pop("rho")
        num_prompt_tokens = inputs.pop("num_prompt_tokens")
        prompt_mask = inputs.pop("prompt_mask")
        clean_input = inputs.pop("clean_inputs")
        pad_token_id = inputs.pop("pad_token_id", None)
        mask_token_id = inputs.pop("mask_token_id", self.mask_token_id)

        if not model.training:
            logits = self.get_logits(model, inputs)
            outputs = model(**inputs) if return_outputs else None
            unscaled_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none", ignore_index=-100
            ).view(logits.shape[0], -1)
            loss = unscaled_loss / t
            loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
            return loss if not return_outputs else (loss, outputs)

        if pad_token_id is None:
            pad_mask = torch.zeros_like(clean_input, dtype=torch.bool)
        else:
            pad_mask = clean_input.eq(pad_token_id)
        response_mask = ~(prompt_mask | pad_mask)
        if not response_mask.any():
            loss = torch.tensor(0.0, device=clean_input.device, requires_grad=True)
            return loss if not return_outputs else (loss, None)

        all_masked_ids = clean_input.masked_fill(response_mask, mask_token_id)
        with torch.no_grad():
            first_logits = self.get_logits(model, {"input_ids": all_masked_ids})
            log_probs = first_logits.log_softmax(dim=-1)
            probs = log_probs.exp()
            entropy = (probs * log_probs).sum(dim=-1).neg().clamp_min(0)
            beta = entropy.sqrt()
            beta_baseline = beta[response_mask].mean().clamp_min(1e-12)
            t_i = 1 - (1 - t).pow(beta / beta_baseline)

            should_mask = torch.bernoulli(t_i).bool() & response_mask
            if not should_mask.any():
                should_mask = labels != -100
            new_labels = clean_input.clone()
            new_labels[~should_mask] = -100
            new_inputs = clean_input.clone()
            new_inputs[should_mask] = mask_token_id

        final_inputs = {"input_ids": new_inputs}
        logits = self.get_logits(model, final_inputs)
        outputs = model(**final_inputs) if return_outputs else None
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), new_labels.view(-1), reduction="none", ignore_index=-100
        ).view(logits.shape[0], -1)
        loss = unscaled_loss / t_i
        denom = should_mask.sum().clamp_min(1)
        loss = loss.sum() / denom

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / denom).item()})
        return loss if not return_outputs else (loss, outputs)
    
    # Vanilla_SFT_LOSS OLD
    def vanilla_sft_loss_old(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        rho = inputs.pop("rho")
        prompt_mask = inputs.pop('prompt_mask')
        clean_input = inputs.pop('clean_inputs')
        inputs.pop("pad_token_id", None)
        inputs.pop("mask_token_id", None)
        logits = self.get_logits(model, inputs)
        outputs = model(**inputs) if return_outputs else None
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss if not return_outputs else (loss, outputs)
    
    # Changed Vanilla loss for mixture.         
    def vanilla_mix_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        rho = inputs.pop("rho")
        prompt_mask = inputs.pop('prompt_mask')
        clean_input = inputs.pop('clean_inputs')
        inputs.pop("pad_token_id", None)
        inputs.pop("mask_token_id", None)
        # outputs = model(**inputs)
        # logits = outputs.logits
        masked = (labels != -100)
        
        if not masked.any():
            loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
            return loss if not return_outputs else (loss, None)
        
        B, N = labels.shape
        t_b = t[:, 0] if t.ndim == 2 else t
        rho_b = rho[:, 0] if rho.ndim == 2 else rho
        t_sup = t_b + rho_b 
        num_masked_sup = masked.sum(dim=1)
        
        frac = t_b / torch.clamp(t_sup, min=1e-8)
        frac = frac.clamp(min=1e-3, max=1.0)
        
        k = (num_masked_sup.float() * frac).round().long()
        k = torch.where(num_masked_sup > 0, k.clamp(min=1), torch.zeros_like(k))
        k = torch.minimum(k, num_masked_sup)         # cannot select more than available
        max_k = int(k.max().item())
        
        rand_scores = torch.rand((B, N), device=labels.device)
        rand_scores = rand_scores.masked_fill(~masked, float("-inf"))
        _, idx = torch.topk(rand_scores, k=max_k, dim=1, largest=True, sorted=False)
        
        take = torch.arange(max_k, device=labels.device)[None, :] < k[:, None]
        keep_mask = torch.zeros_like(masked)
        keep_mask.scatter_(1, idx, take)
        
        drop_mask = masked & ~keep_mask
        
        # Infill dropped masked positions so the input corresponds to the smaller mask (~t)
        modified_input = inputs["input_ids"].clone()
        modified_input[drop_mask] = labels[drop_mask]
        
        new_labels = labels.clone()
        new_labels[drop_mask] = -100
        
        modified_inputs = dict(inputs)
        modified_inputs["input_ids"] = modified_input
        # Create Complementary view, then use for training.
        if self.complementary_training:
            modified_input, new_labels, t = self.build_complimentary_views(
                input_ids=modified_input,
                selected_mask=keep_mask,
                selected_mask_labels=new_labels,
                prompt_mask=prompt_mask,
                clean_input=clean_input,
                t=t
            )
            
            modified_inputs['input_ids'] = modified_input
            clean_input = clean_input.cpu()
            prompt_mask = prompt_mask.cpu()
            B, N = new_labels.shape
        
        outputs = model(**modified_inputs)
        logits = outputs.logits
        
        
        
        V = logits.shape[-1]
        
        unscaled_loss = F.cross_entropy(
            logits.view(-1, V),
            new_labels.view(-1),
            reduction="none",
            ignore_index=-100
        ).view(B, N)
        
        # Complementary Masking on Default does not scale.
        loss = unscaled_loss / t if self.use_time_scaling else unscaled_loss 
        
        denom = (inputs["input_ids"].numel() - num_prompt_tokens)
        loss = loss.sum() / denom
        print(f'Vanilla Loss - {loss}, time - {t.mean(dim=1)} - denom - {denom} - LaVida Loss - {unscaled_loss.sum() / denom}')
        if (self.state.global_step) % self.args.logging_steps == 0:
            self.log({
                      "unscaled_loss": (unscaled_loss.sum() / denom).item(),
                      "loss": loss.item(),
                      "rho": rho[:,0].mean().item(),
                      "t_batch": t[0, 0].mean().item() if self.complementary_training else t[:, 0].mean().item()
                     })
        
        return loss if not return_outputs else (loss, outputs)
    
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        return self.losses[self.loss_mode](model, inputs, num_items_in_batch, return_outputs)
    
    # Loss is same as Bottom-K, with just the order of sorting flipped.
    def top_k(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        return self.bottom_k(model, inputs, num_items_in_batch, return_outputs, top_k=True)
    
         

    def bottom_k(self, model, inputs, num_items_in_batch=None, return_outputs=False, top_k=False):
        labels = inputs.pop("labels")
        t = inputs.pop("t")
        rho = inputs.pop("rho")
        num_prompt_tokens = inputs.pop("num_prompt_tokens")
        prompt_mask = inputs.pop('prompt_mask')
        clean_input = inputs.pop('clean_inputs')
        inputs.pop("pad_token_id", None)
        inputs.pop("mask_token_id", None)
        with torch.set_grad_enabled(self.approximation):
            outputs = model(**inputs)
            logits = outputs.logits
        
        masked_positions = (labels != -100)

        if not masked_positions.any():
            # No masked tokens, return zero loss
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return loss if not return_outputs else (loss, outputs)
        
        B, N, V = logits.shape
        t_b = t[:, 0] if t.ndim == 2 else t # (B,) float
        rho_b = rho[:, 0] if rho.ndim == 2 else rho
        t_sup = t_b + rho_b
        
        frac = t_b / torch.clamp(t_sup, min=1e-3)
        frac = frac.clamp(min=1e-3, max=1.0)
        
        num_masked = masked_positions.sum(dim=1)
        
        # Recovering L*t by dividing by (t+rho)
        k = (num_masked * frac).round().long()
        max_k = int(k.max().item())
        keep_mask = torch.zeros_like(masked_positions) 
        
        
        safe_labels = labels.clone()
        safe_labels[~masked_positions] = 0
        
        
        # Support Top-K vs Bottom-K
        with torch.no_grad():
            log_probs0 = F.log_softmax(logits, dim=-1)
            log_p_gold = log_probs0.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            fill_value = float("-inf") if top_k else float("inf")
            conf = log_p_gold.masked_fill(~masked_positions, fill_value)        
        
            _, idx = torch.topk(conf, k=max_k, dim=1, largest=top_k, sorted=True)
        
            # Keep only the first k_b[b] per example, and get top-mask and bottom-mask.
            take_first_k = torch.arange(max_k, device=masked_positions.device)[None, :] < k[:, None]  # (B, max_k)
            keep_mask.scatter_(1, idx, take_first_k)
        rejected_mask = masked_positions & ~keep_mask
        new_labels = labels.clone()
        new_labels[rejected_mask] = -100
        # Get flat indices of masked positions sorted by confidence
        # num_masked_total = int(masked_positions.sum().item())
        # num_bottom_total = int(bottom_mask.sum().item())
        # num_top_total = int(top_mask.sum().item())
        
        # if self.complementary_training:
        #     modified_input, new_labels, t = self.build_complimentary_views(
        #         input_ids=modified_input,
        #         selected_mask=keep_mask,
        #         selected_mask_labels=new_labels,
        #         prompt_mask=prompt_mask,
        #         clean_input=clean_input,
        #         t=t
        #     )
        #     modified_inputs['input_ids'] = modified_input
        #     clean_input = clean_input.cpu()
        #     prompt_mask = prompt_mask.cpu()
        #     B, N = new_labels.shape
        
        # Fill in stuff, 
        if not self.approximation:
            modified_input = inputs['input_ids'].clone()
            modified_input[rejected_mask] = labels[rejected_mask]
            modified_inputs = {'input_ids': modified_input}
            
            outputs = model(**modified_inputs)
            logits = outputs.logits
            B, N, V = logits.shape
        
        # Create Complementary view, then use for training.
        
        
        # mask_id = self.mask_token_id  # or 126336

        # updated = modified_input.eq(mask_id)
        # old = inputs["input_ids"].eq(mask_id)
        # print(f"Number of Masked Positions After vs Initial - {updated.sum().item()} - {old.sum().item()} and {num_top_total == (old.sum().item() - updated.sum().item())}")
        unscaled_loss = F.cross_entropy(
            logits.view(-1, V), new_labels.view(-1), reduction="none", ignore_index=-100
        ).view(B, N)
        denom = inputs["input_ids"].numel() - num_prompt_tokens
        
        loss = unscaled_loss / t if self.use_time_scaling else unscaled_loss 
        
        # denom = (inputs["input_ids"].numel() - num_prompt_tokens)
        loss = loss.sum() / denom
        # print(f'Loss - {loss}, time - {t.mean(dim=1)} - denom - {denom} - LaVida Loss - {unscaled_loss.sum() / denom}')
        # quit()
        
        # loss =  (unscaled_loss / t).sum() / denom 
        
        if (self.state.global_step) % self.args.logging_steps == 0:
            self.log({
                "unscaled_loss": (unscaled_loss.sum() / denom).item(),
                "loss": loss.item(),
                "rho": rho[:,0].mean().item(),
                "t_batch": t[0, 0].mean().item() if self.complementary_training else t[:, 0].mean().item()
             })
    
        return loss if not return_outputs else (loss, outputs)

class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.pad_token_id = kwargs["tokenizer"].pad_token_id
        self.tokenizer = kwargs["tokenizer"] 
        self.fixed_timestep = kwargs.get("fixed_timestep", False)
        self.timestep_dist = kwargs.get("timestep_dist", None)
        
        self.tokenizer = kwargs["tokenizer"]
        self.rdro_sampling = False
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]
        if self.pad_token_id is None:
            self.pad_token_id = -1
        if "rdro_sampling" in kwargs:
            self.rdro_sampling = kwargs["rdro_sampling"]

    def get_rdro_rho(self, t):
        # If RDRO Sampling is not Enabled, this will make rho as zeros.
        if not self.rdro_sampling:
            return torch.zeros_like(t)
        
        # Sample a 10-25% bigger mask, to get the worst positions. May need tuning later.
        upper = 1.0 - t
        lower = torch.where(upper <= 0.1, torch.zeros_like(upper), torch.full_like(upper, 0.1))
        
        r = torch.rand_like(t)
        rho = lower + (upper - lower)*r 
        return rho     
    
    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            if self.fixed_timestep > 0:
                print(f"Using fixed timestep sampling {self.fixed_timestep}.")
                t = torch.full((B,), self.fixed_timestep, device=input_ids.device)
            elif self.timestep_dist == "discrete_uniform":
                # Discrete timesteps: 1/16, 1/8, 1/4, 1/2
                # These are powers of 2: 2^(-4), 2^(-3), 2^(-2), 2^(-1)
                #print("Using discrete uniform timestep sampling.")
                discrete_timesteps = torch.tensor([1/16, 1/8, 1/4, 1/2, 1/(2**0.5), 1/(2**0.25)], device=input_ids.device)
                # Sample uniformly from these discrete values
                indices = torch.randint(0, len(discrete_timesteps), (B,), device=input_ids.device)
                t = discrete_timesteps[indices]
            else:
                #print("Using continuous uniform timestep sampling.")
                t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]
        
        rho = self.get_rdro_rho(t)
        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)
        rho = rho[:, None].repeat(1,N)
        
        mask_indices = torch.rand((B, N), device=input_ids.device) < (t + rho)
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices, rho

    
    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices, batch["rho"] = self.forward_process(batch)
        batch["pad_token_id"] = self.pad_token_id
        batch["mask_token_id"] = self.mask_token_id
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
            batch['prompt_mask'] = prompt_mask
            batch['clean_inputs'] = batch["input_ids"].clone()
        batch["input_ids"] = noisy_batch.long()
        return batch


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""

class DatasetPreprocessor:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.preprocessors = {
            "simplescaling/s1k": self.preprocess_s1k_dataset,
            "simplescaling/s1K-1.1": self.preprocess_s1k_1_1_dataset,
            "openai/gsm8k": self.preprocess_gsm8k_dataset,
            "divelab/dllm": self.preprocess_divelab_dllm_dataset,
            "KodCode/KodCode-Light-RL-10K": self.preprocess_kodcode
        }
        
    def preprocess_dataset(self, data, tokenizer, max_length, test_split=0.01):
        if self.dataset_name not in self.preprocessors:
            raise ValueError(f"Preprocessor for dataset {self.dataset_name} not implemented.")
        return self.preprocessors[self.dataset_name](data, tokenizer, max_length, test_split)

    def preprocess_s1k_1_1_dataset(self, data, tokenizer, max_length, test_split=0.01):
        preprocessed_data = []
        for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
            question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
            trajectory = f"<reasoning>{data[i]['deepseek_thinking_trajectory']}</reasoning>\n<answer>{data[i]['deepseek_attempt']}</answer>"
            prompt = [{"role": "user", "content": question}]
            response = [{"role": "assistant", "content": trajectory}]
            inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
            tokenized_input = tokenizer(
                inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
            ).input_ids.squeeze(0)
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            preprocessed_data.append(
                {
                    "input_ids": tokenized_input,
                    "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
                }
            )

        random.shuffle(preprocessed_data)
        test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
        train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
        return train_data, test_data

    def preprocess_s1k_dataset(self, data, tokenizer, max_length, test_split=0.01):
        preprocessed_data = []
        for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
            question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
            trajectory = f"<reasoning>{data[i]['thinking_trajectories'][0]}</reasoning>\n<answer>{data[i]['attempt']}</answer>"
            prompt = [{"role": "user", "content": question}]
            response = [{"role": "assistant", "content": trajectory}]
            inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
            tokenized_input = tokenizer(
                inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
            ).input_ids.squeeze(0)
            num_tokens = tokenized_input.shape[0]
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            preprocessed_data.append(
                {
                    "input_ids": tokenized_input,
                    "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
                }
            )

        random.shuffle(preprocessed_data)
        test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
        train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
        return train_data, test_data

    def preprocess_gsm8k_dataset(self, data, tokenizer, max_length, test_split=0.01):
        """
        Preprocess openai/gsm8k data into the same format as preprocess_dataset,
        but using GSM8K's fields: `question` and `answer`.

        We treat everything before '####' as reasoning, and the part after as the final answer.
        If no '####' is present, we just use the whole answer for both.
        """
        preprocessed_data = []
        for i in tqdm(range(len(data)), desc="Preprocessing GSM8K"):
            row = data[i]

            # Build question with your system prompt
            question = SYSTEM_PROMPT + "\n\n" + row["question"]

            raw_answer = row["answer"]

            # Split into reasoning and final answer

            reasoning_part, final_part = raw_answer.split("####", 1)
            reasoning = reasoning_part.strip()
            final_answer = final_part.strip()


            trajectory = (
                f"<reasoning>{reasoning}</reasoning>\n"
                f"<answer>{final_answer}</answer>"
            )

            prompt = [{"role": "user", "content": question}]
            response = [{"role": "assistant", "content": trajectory}]

            # Full input (prompt + answer trajectory)
            inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)

            # Prompt-only string to get prompt length
            prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"

            tokenized_input = tokenizer(
                inputs,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            ).input_ids.squeeze(0)

            tokenized_prompt = tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            
            preprocessed_data.append(
                {
                    "input_ids": tokenized_input,
                    "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
                }
            )

        random.shuffle(preprocessed_data)
        test_size = int(len(preprocessed_data) * test_split)
        test_data = preprocessed_data[:test_size]
        train_data = preprocessed_data[test_size:]
        return train_data, test_data
    
    def preprocess_divelab_dllm_dataset(self, data, tokenizer, max_length, test_split=0.01):
        """
        Preprocess divelab/dllm data using the repacked schema.
        Fields: 'sol', 'question', 'thinking_trajectories', 'source_type'
        """
        preprocessed_data = []
        
        for i in tqdm(range(len(data)), desc="Preprocessing divelab/dllm"):
            row = data[i]
            
            # 1. Build the question with system prompt
            question = SYSTEM_PROMPT + "\n\n" + row["question"]

            # 2. Extract reasoning (thinking_trajectories is list[str])
            # We take the first trajectory if available
            thoughts = row["thinking_trajectories"]
            thought = thoughts[0] if isinstance(thoughts, list) and len(thoughts) > 0 else ""

            # 3. Format the trajectory with reasoning and answer tags
            # 'sol' contains the ground truth answer
            trajectory = (
                f"<reasoning>{thought}</reasoning>\n"
                f"<answer>{row['sol']}</answer>"
            )
            prompt = [{"role": "user", "content": question}]
            response = [{"role": "assistant", "content": trajectory}]

            # 4. Tokenization and Chat Template application
            inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
            prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"

            tokenized_input = tokenizer(
                inputs,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            ).input_ids.squeeze(0)

            decoded = tokenizer.decode(
                tokenized_input,
                skip_special_tokens=False,  # set True if you want to hide [PAD]/[BOS]/[EOS]
            )
            # print(decoded)

            tokenized_prompt = tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            preprocessed_data.append(
                {
                    "input_ids": tokenized_input,
                    "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
                }
            )

        # Shuffle and split
        random.shuffle(preprocessed_data)
        test_size = int(len(preprocessed_data) * test_split)
        test_data = preprocessed_data[:test_size]
        train_data = preprocessed_data[test_size:]
        return train_data, test_data
    
    def _extract_kodcode_prompt_and_response(self, row):
        """
        Extract prompt and response from a KodCode row.

        Returns:
            prompt_text: user-side coding question
            response_text: assistant-side text formatted as
                <reasoning>...</reasoning>
                <answer>...</answer>
        """

        question = row.get("question", "").strip()
        conversations = row.get("conversations", None)

        user_msg = None
        assistant_msg = None

        if isinstance(conversations, list):
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                role = turn.get("from", "")
                value = turn.get("value", "")
                if role == "human" and user_msg is None:
                    user_msg = value.strip()
                elif role == "gpt" and assistant_msg is None:
                    assistant_msg = value.strip()

        prompt_text = user_msg if user_msg else question

        # Prefer the full assistant response if present, because it may include reasoning.
        if assistant_msg:
            text = assistant_msg.strip()

            # Normalize <think> -> <reasoning>
            text = text.replace("<think>", "<reasoning>")
            text = text.replace("</think>", "</reasoning>")

            has_reasoning = "<reasoning>" in text and "</reasoning>" in text
            has_answer = "<answer>" in text and "</answer>" in text

            if has_reasoning and has_answer:
                response_text = text
            elif has_reasoning and not has_answer:
                # If reasoning exists but answer tag doesn't, wrap fallback code in <answer>
                answer_code = row.get("r1_solution", "") or row.get("solution", "")
                response_text = f"{text}\n<answer>{answer_code.strip()}</answer>"
            else:
                # No XML structure in assistant text; use it as reasoning, code as answer
                answer_code = row.get("r1_solution", "") or row.get("solution", "")
                response_text = (
                    f"<reasoning>{text}</reasoning>\n"
                    f"<answer>{answer_code.strip()}</answer>"
                )
        else:
            # Fallback: no assistant conversation, so use empty reasoning + solution
            answer_code = row.get("r1_solution", "") or row.get("solution", "")
            response_text = (
                f"<reasoning></reasoning>\n"
                f"<answer>{answer_code.strip()}</answer>"
            )

        return prompt_text, response_text


    def preprocess_kodcode(self, data, tokenizer, max_length, test_split=0.01):
        """
        Preprocess KodCode/KodCode-Light-RL-10K into the same format as your SFT pipeline.
        """
        preprocessed_data = []

        for i in tqdm(range(len(data)), desc="Preprocessing KodCode-Light-RL-10K"):
            row = data[i]

            prompt_text, response_text = self._extract_kodcode_prompt_and_response(row)

            # Your trainer format
            question = SYSTEM_PROMPT + "\n\n" + prompt_text

            prompt = [{"role": "user", "content": question}]
            response = [{"role": "assistant", "content": response_text}]

            full_text = tokenizer.apply_chat_template(prompt + response, tokenize=False)
            prompt_only = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"

            tokenized_input = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            ).input_ids.squeeze(0)

            tokenized_prompt = tokenizer(
                prompt_only,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            preprocessed_data.append(
                {
                    "input_ids": tokenized_input,
                    "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
                }
            )

        random.shuffle(preprocessed_data)
        test_size = int(len(preprocessed_data) * test_split)
        test_data = preprocessed_data[:test_size]
        train_data = preprocessed_data[test_size:]
        return train_data, test_data
        
"""   
Steps

1. Do Vanilla stuff
2. Then Create the Complement.
3. Concat.
4. a. Calculate Numerator - 1/t*CE_Masked (Scaled)
4. b. Calculate Non-Scaled Numerator - CE_Masked (UnScaled)
5. Calculate Denominator - D
6. Loss = (Num_1 + Num_2) / (Denom_1 + Denom_2) 
    
"""
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_dataset
    
    dataset_name = "KodCode/KodCode-Light-RL-10K"
    tokenizer_name = "GSAI-ML/LLaDA-8B-Instruct"
    max_length = 2048
    num_examples = 8

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Loading dataset sample...")
    ds = load_dataset(dataset_name, split=f"train[:{num_examples}]")

    print("Building preprocessor...")
    preprocessor = DatasetPreprocessor(dataset_name)

    print("Running preprocessing...")
    train_data, test_data = preprocessor.preprocess_dataset(
        ds,
        tokenizer=tokenizer,
        max_length=max_length,
        test_split=0.25,
    )

    print(f"Train size: {len(train_data)}")
    print(f"Test size:  {len(test_data)}")

    if len(train_data) > 0:
        sample = train_data[0]
        print("\n=== Sample processed item ===")
        print("input_ids shape:", sample["input_ids"].shape)
        print("prompt_lengths:", sample["prompt_lengths"])

        decoded = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        print("\n=== Decoded sample ===")
        print(decoded)

    print("\nDone.")
