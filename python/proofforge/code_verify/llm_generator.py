"""LLM inference wrappers for code generation (App 1).

Provides pluggable generate_fn implementations for CodeVerificationPipeline.
Two backends: Ollama (local, no GPU) and HuggingFace (GPU, GRPO-compatible).
"""

from __future__ import annotations

from typing import Callable


def create_ollama_generator(
    model: str = "deepseek-coder:7b-instruct",
    base_url: str = "http://127.0.0.1:11434",
) -> Callable:
    """Create a generate_fn using Ollama's HTTP API.

    Good for evaluation and testing. Does NOT support log-probability
    computation needed for GRPO — use HuggingFace backend for training.

    Returns:
        generate_fn(prompt, n, temperature, max_tokens) -> list[str]
    """
    import requests

    def generate(prompt: str, n: int, temperature: float, max_tokens: int) -> list[str]:
        solutions = []
        for _ in range(n):
            try:
                resp = requests.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "stop": ["\n\n\n", "```"],
                        },
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                solutions.append(resp.json().get("response", ""))
            except Exception as e:
                solutions.append(f"# generation error: {e}")
        return solutions

    return generate


def create_hf_generator(model, tokenizer, device="cuda") -> Callable:
    """Create a generate_fn from a loaded HuggingFace model + tokenizer.

    The model must already be loaded (with LoRA if applicable).
    Supports the same interface as grpo_lean_reward.py generate_completions.

    Returns:
        generate_fn(prompt, n, temperature, max_tokens) -> list[str]
    """
    import torch

    def generate(prompt: str, n: int, temperature: float, max_tokens: int) -> list[str]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        solutions = []
        for _ in range(n):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            solutions.append(text)
        return solutions

    return generate
