#!/bin/bash
# ProofForge Phase 4: Install ML dependencies for neural GRPO training
# Requires Python 3.10+ and pip

set -e

echo "=== ProofForge ML Stack Setup ==="

# Core ML
pip3 install torch torchvision torchaudio

# Hugging Face ecosystem
pip3 install transformers accelerate peft

# TRL for GRPO training
pip3 install trl

# Fast inference (optional, for generation)
pip3 install vllm 2>/dev/null || echo "vLLM not available on this platform (needs Linux + CUDA)"

# HTTP server for Rust bridge
pip3 install flask

# Utilities
pip3 install datasets sentencepiece protobuf

echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  MPS:  {torch.backends.mps.is_available()}')

import transformers
print(f'Transformers {transformers.__version__}')

import trl
print(f'TRL {trl.__version__}')

import peft
print(f'PEFT {peft.__version__}')

print()
print('✓ ProofForge ML stack ready')
"
