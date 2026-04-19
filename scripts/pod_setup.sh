#!/bin/bash
# ProofForge Pod Setup — downloads model + compiles flash-attn BEFORE training.
# Run this ONCE on a fresh pod. On restart, pip packages are reinstalled fast
# from pip cache, but model weights and flash-attn wheel are cached on /workspace.
#
# Usage:
#   bash scripts/pod_setup.sh          # setup only
#   bash scripts/pod_setup.sh --launch  # setup then launch training

set -euo pipefail

WORKSPACE="/workspace"
HF_CACHE="$WORKSPACE/hf_cache"
ELAN_HOME="$HOME/.elan"
REPO_DIR="$WORKSPACE/proofforge"
FA_WHEEL_DIR="$WORKSPACE/wheels"
MODEL="deepseek-ai/DeepSeek-Prover-V2-7B"

export PATH="$ELAN_HOME/bin:$PATH"
export HF_HOME="$HF_CACHE"

echo "============================================================"
echo "  ProofForge Pod Setup"
echo "============================================================"
echo ""

# -------------------------------------------------------------------
# Step 1: Install pip packages (uses system pip — fast on restart via pip cache)
# -------------------------------------------------------------------
echo "=== Step 1: Pip packages ==="

if python3 -c "import transformers, peft, accelerate, bitsandbytes, sentencepiece" 2>/dev/null; then
    echo "  Already installed."
else
    echo "  Installing..."
    pip install transformers peft accelerate bitsandbytes sentencepiece protobuf -q 2>&1 | tail -1
fi

pip install -e "$REPO_DIR/python/" -q 2>&1 | tail -1

python3 -c "
import torch, transformers, peft
print('  torch:', torch.__version__)
print('  transformers:', transformers.__version__)
print('  peft:', peft.__version__)
print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"
echo ""

# -------------------------------------------------------------------
# Step 2: Download model weights (cached on /workspace)
# -------------------------------------------------------------------
echo "=== Step 2: Model weights ==="

MODEL_DIR="$HF_CACHE/hub/models--deepseek-ai--DeepSeek-Prover-V2-7B"
if [ -d "$MODEL_DIR/snapshots" ] && [ "$(ls -A "$MODEL_DIR/snapshots/" 2>/dev/null)" ]; then
    N_FILES=$(find "$MODEL_DIR/snapshots/" -name "*.safetensors" 2>/dev/null | wc -l)
    echo "  CACHED — $N_FILES safetensors files"
else
    echo "  Downloading $MODEL..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', cache_dir='$HF_CACHE')
print('  Download complete.')
"
fi
echo ""

# -------------------------------------------------------------------
# Step 3: Lean 4 (cached on /workspace)
# -------------------------------------------------------------------
echo "=== Step 3: Lean 4 ==="

if [ -x "$ELAN_HOME/bin/lean" ]; then
    echo "  CACHED — $("$ELAN_HOME/bin/lean" --version 2>/dev/null | head -1)"
else
    echo "  Installing..."
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
        | sh -s -- -y --default-toolchain leanprover/lean4:v4.8.0 2>&1 | tail -2
    echo "  Installed: $("$ELAN_HOME/bin/lean" --version 2>/dev/null | head -1)"
fi
echo ""

# -------------------------------------------------------------------
# Step 4: Flash Attention 2 (wheel cached on /workspace)
# -------------------------------------------------------------------
echo "=== Step 4: Flash Attention 2 ==="

if python3 -c "import flash_attn; print('  CACHED —', flash_attn.__version__)" 2>/dev/null; then
    true
else
    # Check for cached wheel
    mkdir -p "$FA_WHEEL_DIR"
    FA_WHEEL=$(find "$FA_WHEEL_DIR" -name "flash_attn*.whl" 2>/dev/null | head -1)
    if [ -n "$FA_WHEEL" ]; then
        echo "  Installing from cached wheel: $FA_WHEEL"
        pip install "$FA_WHEEL" -q 2>&1 | tail -1
    else
        echo "  Compiling (one-time, ~5-10 min)..."
        pip install flash-attn --no-build-isolation 2>&1 | tail -3
        # Cache the wheel for future restarts
        pip download flash-attn --no-build-isolation --no-deps -d "$FA_WHEEL_DIR" 2>&1 | tail -1 || true
    fi
    python3 -c "import flash_attn; print('  Installed:', flash_attn.__version__)"
fi
echo ""

# -------------------------------------------------------------------
# Step 5: Verification
# -------------------------------------------------------------------
echo "=== Step 5: Verification ==="

ERRORS=0

python3 -c "import torch; assert torch.cuda.is_available(); print('  [OK] GPU:', torch.cuda.get_device_name(0))" || { echo "  [FAIL] No GPU"; ERRORS=$((ERRORS+1)); }

python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$MODEL', cache_dir='$HF_CACHE', trust_remote_code=True)
print('  [OK] Tokenizer loads, vocab:', t.vocab_size)
" || { echo "  [FAIL] Tokenizer"; ERRORS=$((ERRORS+1)); }

python3 -c "import flash_attn; print('  [OK] flash_attn:', flash_attn.__version__)" || echo "  [WARN] flash_attn missing (will use SDPA)"

"$ELAN_HOME/bin/lean" --version > /dev/null 2>&1 && echo "  [OK] Lean 4" || { echo "  [FAIL] Lean"; ERRORS=$((ERRORS+1)); }

python3 -c "from proofforge.controller.phase_detector import PhaseDetector; print('  [OK] proofforge')" || { echo "  [FAIL] proofforge"; ERRORS=$((ERRORS+1)); }

[ -f "$REPO_DIR/data/curriculum_radical.json" ] && echo "  [OK] curriculum" || { echo "  [FAIL] curriculum"; ERRORS=$((ERRORS+1)); }
[ -f "$REPO_DIR/data/lean_definitions.lean" ] && echo "  [OK] lean_definitions" || { echo "  [FAIL] lean_definitions"; ERRORS=$((ERRORS+1)); }

python3 -c "compile(open('$REPO_DIR/scripts/grpo_lean_reward.py').read(),'<>','exec'); print('  [OK] script compiles')" || { echo "  [FAIL] script"; ERRORS=$((ERRORS+1)); }

echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo "  SETUP FAILED — $ERRORS errors."
    exit 1
fi
echo "  ALL CHECKS PASSED"
echo ""

# -------------------------------------------------------------------
# Step 6: Optional launch
# -------------------------------------------------------------------
if [ "${1:-}" = "--launch" ]; then
    echo "=== Launching training ==="

    cd "$REPO_DIR"
    export PATH="$ELAN_HOME/bin:$PATH"
    export HF_HOME="$HF_CACHE"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
    PROOFFORGE_LEAN_DEFS="$REPO_DIR/data/lean_definitions.lean" \
    nohup python3 -u scripts/grpo_lean_reward.py \
      --model "$MODEL" \
      --stage both \
      --n-completions-per-theorem 200 \
      --sft-epochs 2 \
      --sft-lr 2e-5 \
      --steps 200 \
      --group-size 32 \
      --max-new-tokens 128 \
      --max-theorems 98 \
      --theorems-path "$REPO_DIR/data/curriculum_radical.json" \
      --checkpoint-schedule "0,5,10,20,30,50,75,100,150,200" \
      --temperatures 0.3 0.7 1.2 \
      --lean-workers 16 \
      --learning-rate 5e-6 \
      --min-lr-ratio 0.1 \
      --enable-efficiency \
      --efficiency-phase-in-step 50 \
      --efficiency-phase-in-rate 0.20 \
      --enable-controller \
      --controller-bias-correction 0.15 \
      --temp-start 1.0 \
      --temp-end 0.5 \
      --output-dir /workspace/grpo_round3d \
      > /workspace/round3d_stdout.log 2>&1 &

    echo $! > /workspace/round3d_pid.txt
    echo "  PID: $(cat /workspace/round3d_pid.txt)"
    sleep 2
    if ps -p $(cat /workspace/round3d_pid.txt) > /dev/null 2>&1; then
        echo "  RUNNING"
    else
        echo "  FAILED"
        tail -10 /workspace/round3d_stdout.log
        exit 1
    fi
else
    echo "Run with --launch to start training."
fi
