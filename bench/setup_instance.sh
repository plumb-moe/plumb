#!/bin/bash
# Instance setup script for plumb 3-way placement benchmark.
# Run once after SSH'ing into the Vast.ai instance.
set -euo pipefail

echo "=== plumb bench instance setup ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- system deps ---
apt-get update -qq && apt-get install -y -qq git curl tmux htop 2>/dev/null || true

# --- clone plumb repo ---
if [ ! -d /workspace/plumb ]; then
  git clone https://github.com/plumb-moe/plumb.git /workspace/plumb
fi
cd /workspace/plumb
git pull --rebase

# --- install plumb package ---
pip install -e packages/plumb -q

# --- install vllm if not already present ---
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || \
  pip install vllm -q

# --- install autoawq for AWQ quantized models ---
pip install autoawq -q 2>/dev/null || echo "autoawq install failed — will use non-AWQ model"

# --- install other deps ---
pip install safetensors requests numpy -q

# --- verify ---
echo ""
echo "=== Versions ==="
python3 -c "import vllm; print('vllm:', vllm.__version__)"
python3 -c "import plumb; print('plumb: ok')"
python3 -c "import safetensors; print('safetensors: ok')"
python3 -c "import autoawq; print('autoawq: ok')" 2>/dev/null || echo "autoawq: not available"

echo ""
echo "=== Setup complete. Run the benchmark with: ==="
echo "  cd /workspace/plumb && python bench/benchmark_placement_3way.py --tp 4 --num-gpus 4"
