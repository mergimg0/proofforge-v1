#!/bin/bash
# H100 SXM RunPod — GRPO Training + Hidden State Collection
#
# This script sets up and runs the full instrumented training pipeline.
# Upload to the pod and run: bash h100_deploy.sh
#
# Expected runtime: ~6-8 hours on H100 SXM
# Expected output: ~3000 trajectory .npz files + checkpoint weights

set -e

echo "============================================================"
echo "CGLE Phase 2: Instrumented GRPO Training"
echo "============================================================"

# 1. Install dependencies
echo "[1/5] Installing dependencies..."
pip install transformers accelerate sentencepiece protobuf trl peft datasets 2>&1 | tail -3

# 2. Set up HuggingFace cache on persistent volume
export HF_HOME=/workspace/hf_cache
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p $HF_HOME

# 3. Set up data directory
mkdir -p /workspace/proofforge/data
mkdir -p /workspace/grpo_run

# Check if theorems.json exists
if [ ! -f /workspace/proofforge/data/theorems.json ]; then
    echo "ERROR: theorems.json not found!"
    echo "Upload it: scp theorems.json root@HOST:/workspace/proofforge/data/"
    exit 1
fi

# Check if training script exists
if [ ! -f /workspace/grpo_instrumented.py ]; then
    echo "ERROR: grpo_instrumented.py not found!"
    echo "Upload it: scp grpo_instrumented.py root@HOST:/workspace/"
    exit 1
fi

echo "[2/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[3/5] Starting instrumented GRPO training..."
echo "  Checkpoints: 0,5,10,20,30,50,75,100,150,200"
echo "  Per checkpoint: 50 theorems x 3 temps x 1 attempt = 150 trajectories"
echo "  Total trajectories: ~1500"
echo ""

# 4. Run training
cd /workspace
python3 grpo_instrumented.py \
    --model deepseek-ai/DeepSeek-Prover-V2-7B \
    --steps 200 \
    --group-size 8 \
    --output-dir /workspace/grpo_run \
    --checkpoint-schedule "0,5,10,20,30,50,75,100,150,200" \
    --max-new-tokens 256 \
    2>&1 | tee /workspace/grpo_training.log

echo ""
echo "[4/5] Training complete. Checking output..."
echo "Checkpoint directories:"
ls -d /workspace/grpo_run/checkpoint_*/ 2>/dev/null | head -15
echo ""
echo "Total .npz files:"
find /workspace/grpo_run -name "*.npz" | wc -l
echo ""
echo "Disk usage:"
du -sh /workspace/grpo_run/

echo ""
echo "[5/5] Done. Download with:"
echo "  scp -r root@HOST:/workspace/grpo_run/ ~/projects/proofforge/grpo_run/"
echo "============================================================"
