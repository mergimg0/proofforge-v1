"""GRPO training logic — model loading, gradient computation, log-probability.

Extracted from train_server.py to reduce monolith re-read cost.
train_server.py imports and calls these functions from Flask endpoints.
"""

from __future__ import annotations

import copy
from typing import Optional

# Defer ML imports
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    pass


def get_device(config) -> str:
    """Select best available device."""
    if config.device != "auto":
        return config.device
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def load_model(config, state):
    """Load model with LoRA adapters. Returns (model, ref_model, tokenizer, optimizer, scheduler)."""
    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch not installed. Running in mock mode.")
        state.model_loaded = False
        return None, None, None, None, None

    device = get_device(config)
    print(f"Loading {config.model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to("mps")

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.train(False)
    print("Reference model frozen for KL penalty.")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=config.max_training_steps,
    )

    state.model_loaded = True
    print(f"Model loaded on {device}. LoRA adapters attached. Optimizer ready.")
    return model, ref_model, tokenizer, optimizer, scheduler


def compute_completion_log_prob(model, tokenizer, prompt, completion_token_ids, prompt_ids=None):
    """Compute log-probability of completion tokens under the current policy."""
    if prompt_ids is None:
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    completion_tensor = torch.tensor([completion_token_ids], dtype=torch.long, device=model.device)
    full_ids = torch.cat([prompt_ids, completion_tensor], dim=1)
    outputs = model(input_ids=full_ids, use_cache=False)
    logits = outputs.logits
    prompt_len = prompt_ids.shape[1]
    comp_len = len(completion_token_ids)
    if comp_len == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)
    pred_logits = logits[0, prompt_len - 1 : prompt_len + comp_len - 1, :]
    targets = completion_tensor[0]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs[torch.arange(comp_len, device=model.device), targets]
    return token_log_probs.sum()



# grpo_step() was removed — it was dead code (imported as _grpo_step_impl in
# train_server.py but never called). The active GRPO step with PPO ratio
# clipping lives in train_server.py:_grpo_step().
