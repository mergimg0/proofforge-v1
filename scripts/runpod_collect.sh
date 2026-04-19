#!/bin/bash
# RunPod A40 — CGLE Phase 2 Hidden State Collection
# Copy-paste this entire script into the pod terminal.
#
# Prerequisites: RunPod Pytorch 2.4.0 template (has torch + CUDA)
# Expected runtime: ~2-3 hours for full collection

set -e

echo "=== CGLE Phase 2: Setting up collection pipeline ==="

# 1. Install additional Python deps (torch is already in the template)
pip install transformers numpy accelerate sentencepiece protobuf 2>&1 | tail -3

# 2. Set HuggingFace cache to the persistent volume (survives pod restarts)
export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME

# 3. Create the collection scripts directly on the pod
mkdir -p /workspace/proofforge/python/cgle_analysis
mkdir -p /workspace/proofforge/data
mkdir -p /workspace/cgle_data

# 4. Download the theorem dataset
cat > /workspace/proofforge/data/theorems.json << 'THEOREMS_EOF'
{
  "description": "ProofForge toy experiment theorem dataset",
  "version": "0.1.0",
  "tiers": [
    {
      "name": "trivial",
      "description": "One-tactic proofs",
      "theorems": [
        {"id": "t1_01", "statement": "theorem pf_t1_01 : True := by", "known_proof": "trivial", "difficulty": 1},
        {"id": "t1_02", "statement": "theorem pf_t1_02 : 1 = 1 := by", "known_proof": "rfl", "difficulty": 1},
        {"id": "t1_03", "statement": "theorem pf_t1_03 : 2 + 3 = 5 := by", "known_proof": "decide", "difficulty": 1},
        {"id": "t1_04", "statement": "theorem pf_t1_04 : 10 * 10 = 100 := by", "known_proof": "decide", "difficulty": 1},
        {"id": "t1_05", "statement": "theorem pf_t1_05 : 0 = 0 := by", "known_proof": "rfl", "difficulty": 1},
        {"id": "t1_06", "statement": "theorem pf_t1_06 : Nat.succ 0 = 1 := by", "known_proof": "rfl", "difficulty": 1},
        {"id": "t1_07", "statement": "theorem pf_t1_07 : (3 : Nat) < 5 := by", "known_proof": "decide", "difficulty": 1},
        {"id": "t1_08", "statement": "theorem pf_t1_08 : (2 : Nat) ≤ 2 := by", "known_proof": "decide", "difficulty": 1},
        {"id": "t1_09", "statement": "theorem pf_t1_09 : ¬ False := by", "known_proof": "trivial", "difficulty": 1},
        {"id": "t1_10", "statement": "theorem pf_t1_10 : True ∧ True := by", "known_proof": "exact ⟨trivial, trivial⟩", "difficulty": 1}
      ]
    },
    {
      "name": "simple_tactic",
      "description": "Needs simp, omega, or intro",
      "theorems": [
        {"id": "t2_01", "statement": "theorem pf_t2_01 (n : Nat) : n = n := by", "known_proof": "rfl", "difficulty": 2},
        {"id": "t2_02", "statement": "theorem pf_t2_02 (n : Nat) : 0 + n = n := by", "known_proof": "simp", "difficulty": 2},
        {"id": "t2_03", "statement": "theorem pf_t2_03 (n : Nat) : n + 0 = n := by", "known_proof": "simp", "difficulty": 2},
        {"id": "t2_04", "statement": "theorem pf_t2_04 (a b : Nat) : a + b = b + a := by", "known_proof": "omega", "difficulty": 2},
        {"id": "t2_05", "statement": "theorem pf_t2_05 (n : Nat) : n * 1 = n := by", "known_proof": "simp", "difficulty": 2},
        {"id": "t2_06", "statement": "theorem pf_t2_06 (n : Nat) : 1 * n = n := by", "known_proof": "simp", "difficulty": 2},
        {"id": "t2_07", "statement": "theorem pf_t2_07 (p : Prop) : p → p := by", "known_proof": "intro h; exact h", "difficulty": 2},
        {"id": "t2_08", "statement": "theorem pf_t2_08 (p q : Prop) : p → q → p ∧ q := by", "known_proof": "intro hp hq; exact ⟨hp, hq⟩", "difficulty": 2},
        {"id": "t2_09", "statement": "theorem pf_t2_09 (p q : Prop) : p ∧ q → p := by", "known_proof": "intro h; exact h.1", "difficulty": 2},
        {"id": "t2_10", "statement": "theorem pf_t2_10 (p q : Prop) : p ∧ q → q := by", "known_proof": "intro h; exact h.2", "difficulty": 2}
      ]
    },
    {
      "name": "multi_step",
      "description": "Requires multiple tactics",
      "theorems": [
        {"id": "t3_01", "statement": "theorem pf_t3_01 (p q : Prop) (hp : p) (hq : q) : p ∧ q := by", "known_proof": "exact ⟨hp, hq⟩", "difficulty": 3},
        {"id": "t3_02", "statement": "theorem pf_t3_02 (p q : Prop) : p ∨ q → q ∨ p := by", "known_proof": "intro h; cases h with | inl hp => right; exact hp | inr hq => left; exact hq", "difficulty": 3},
        {"id": "t3_03", "statement": "theorem pf_t3_03 (n : Nat) : n + 1 > 0 := by", "known_proof": "omega", "difficulty": 3},
        {"id": "t3_04", "statement": "theorem pf_t3_04 (a b c : Nat) : (a + b) + c = a + (b + c) := by", "known_proof": "omega", "difficulty": 3},
        {"id": "t3_05", "statement": "theorem pf_t3_05 (n : Nat) : 0 ≤ n := by", "known_proof": "omega", "difficulty": 3}
      ]
    }
  ]
}
THEOREMS_EOF

echo "Theorem dataset written."

# 5. Write the collection script
cat > /workspace/proofforge/python/cgle_analysis/__init__.py << 'EOF'
"""CGLE Substrate Analysis — Phase 2 empirical validation."""
EOF

# The main collect.py will be copied via scp or written inline
# For now, check if it exists:
if [ ! -f /workspace/proofforge/python/cgle_analysis/collect.py ]; then
    echo ""
    echo "ERROR: collect.py not found!"
    echo "Upload it via: scp collect.py root@<pod-ip>:/workspace/proofforge/python/cgle_analysis/"
    echo "Or paste it into the Jupyter notebook."
    exit 1
fi

echo ""
echo "=== Setup complete. Starting collection... ==="
echo ""

# 6. Run the collection
cd /workspace/proofforge/python
python3 -m cgle_analysis.collect \
    --model deepseek-ai/DeepSeek-Prover-V2-7B \
    --output-dir /workspace/cgle_data \
    --temperatures 0.3 0.7 1.2 \
    --attempts 2 \
    --max-tokens 512 \
    --skip-verify

echo ""
echo "=== Collection complete! ==="
echo "Data saved to /workspace/cgle_data/"
echo "Download with: scp -r root@<pod-ip>:/workspace/cgle_data/ ~/projects/proofforge/cgle_data/"
