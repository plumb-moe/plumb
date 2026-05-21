#!/usr/bin/env bash
# vast_bench_run.sh — one-shot Mixtral-8x7B ShareGPT benchmark on Vast.ai
# Usage: bench/vast_bench_run.sh [--model MODEL] [--tp TP] [--dry-run]
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
TOKENIZER=""
TOKENIZER_MODE="mistral"
TP=4
DRY_RUN=false
VAST_API_KEY="${VAST_API_KEY:-}"
if [[ -z "$VAST_API_KEY" ]]; then
  echo "ERROR: VAST_API_KEY env var not set. Export it before running:" >&2
  echo "  export VAST_API_KEY=<your-key>" >&2
  echo "  Get your key at: https://cloud.vast.ai/account/" >&2
  exit 1
fi
INSTANCE_ID=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2";          shift 2 ;;
    --tokenizer)      TOKENIZER="$2";      shift 2 ;;
    --tokenizer-mode) TOKENIZER_MODE="$2"; shift 2 ;;
    --tp)             TP="$2";             shift 2 ;;
    --dry-run) DRY_RUN=true; shift   ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Dry-run wrapper — prints the command instead of executing it
# ---------------------------------------------------------------------------
run() {
  if $DRY_RUN; then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# EXIT trap — destroy the instance if the script exits unexpectedly
# ---------------------------------------------------------------------------
cleanup() {
  local exit_code=$?
  if [[ -n "$INSTANCE_ID" && "$exit_code" -ne 0 ]]; then
    echo ""
    echo "Script exiting with code $exit_code — destroying instance $INSTANCE_ID to avoid runaway billing..."
    run vastai destroy instance "$INSTANCE_ID" --api-key "$VAST_API_KEY" || true
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1. Prerequisites check
# ---------------------------------------------------------------------------
echo "=== Checking prerequisites ==="
MISSING=()
for tool in vastai jq ssh scp; do
  if ! command -v "$tool" &>/dev/null; then
    MISSING+=("$tool")
  fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "ERROR: The following required tools are missing from PATH:" >&2
  for t in "${MISSING[@]}"; do
    case "$t" in
      vastai) echo "  vastai  — install with: pip install vastai" >&2 ;;
      jq)     echo "  jq      — install with: apt install jq  OR  brew install jq" >&2 ;;
      ssh)    echo "  ssh     — install openssh-client" >&2 ;;
      scp)    echo "  scp     — install openssh-client" >&2 ;;
    esac
  done
  exit 1
fi
echo "All prerequisites found."

# ---------------------------------------------------------------------------
# 2. List and destroy idle instances
# ---------------------------------------------------------------------------
echo ""
echo "=== Checking for existing Vast.ai instances ==="
INSTANCES_JSON=$(vastai show instances --api-key "$VAST_API_KEY" --raw 2>/dev/null || echo "[]")

INSTANCE_COUNT=$(echo "$INSTANCES_JSON" | jq 'length')

if [[ "$INSTANCE_COUNT" -gt 0 ]]; then
  echo "Found $INSTANCE_COUNT existing instance(s):"
  echo "$INSTANCES_JSON" | jq -r '.[] | "  id=\(.id)  status=\(.actual_status // "unknown")  cost=\(.dph_total // 0 | . * 100 | round / 100)/hr  label=\(.label // "-")"'
  echo ""
  read -r -p "Destroy all $INSTANCE_COUNT instance(s)? [y/N] " CONFIRM_DESTROY
  if [[ "${CONFIRM_DESTROY,,}" == "y" ]]; then
    echo "$INSTANCES_JSON" | jq -r '.[].id' | while read -r iid; do
      echo "  Destroying instance $iid..."
      run vastai destroy instance "$iid" --api-key "$VAST_API_KEY"
    done
    echo "All existing instances destroyed."
  else
    echo "Keeping existing instances. Proceeding to find a new offer."
  fi
else
  echo "No existing instances found."
fi

# ---------------------------------------------------------------------------
# 3. Find best offer
# ---------------------------------------------------------------------------
echo ""
echo "=== Searching for GPU offers (4x GPU, >=10GB VRAM each, sub \$0.55/hr) ==="
OFFERS_JSON=$(vastai search offers \
  'num_gpus=4 gpu_ram>=10 cuda_vers>=11.8 disk_space>=150 dph<=0.55' \
  --order dph_total --raw 2>/dev/null || echo "[]")

OFFER_COUNT=$(echo "$OFFERS_JSON" | jq 'length')
if [[ "$OFFER_COUNT" -eq 0 ]]; then
  echo "ERROR: No matching offers found. Try relaxing constraints." >&2
  exit 1
fi

echo "Top 5 offers:"
echo "$OFFERS_JSON" | jq -r '
  .[0:5][] |
  "  offer_id=\(.id)  gpu=\(.gpu_name)  vram=\(.gpu_ram)GB  cost=\(.dph_total | . * 100 | round / 100)/hr  loc=\(.geolocation // "unknown")"
'

OFFER_ID=$(echo "$OFFERS_JSON" | jq -r '.[0].id')
OFFER_GPU=$(echo "$OFFERS_JSON" | jq -r '.[0].gpu_name')
OFFER_VRAM=$(echo "$OFFERS_JSON" | jq -r '.[0].gpu_ram')
OFFER_COST=$(echo "$OFFERS_JSON" | jq -r '.[0].dph_total | . * 100 | round / 100')
OFFER_LOC=$(echo "$OFFERS_JSON" | jq -r '.[0].geolocation // "unknown"')

echo ""
echo "Selected offer: $OFFER_GPU x2 (${OFFER_VRAM}GB VRAM each) @ \$$OFFER_COST/hr — $OFFER_LOC"
echo ""
read -r -p "Rent this instance? [y/N] " CONFIRM_RENT
if [[ "${CONFIRM_RENT,,}" != "y" ]]; then
  echo "Aborted by user."
  exit 0
fi

# ---------------------------------------------------------------------------
# 4. Rent instance
# ---------------------------------------------------------------------------
echo ""
echo "=== Renting instance ==="
if $DRY_RUN; then
  echo "[DRY-RUN] vastai create instance $OFFER_ID --image pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime --disk 200 --api-key \$VAST_API_KEY --raw"
  INSTANCE_ID="DRY-RUN-ID"
else
  CREATE_RESP=$(vastai create instance "$OFFER_ID" \
    --image pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime \
    --disk 200 \
    --api-key "$VAST_API_KEY" \
    --raw)
  INSTANCE_ID=$(echo "$CREATE_RESP" | jq -r '.new_contract')
  if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "null" ]]; then
    echo "ERROR: Failed to rent instance. API response:" >&2
    echo "$CREATE_RESP" >&2
    exit 1
  fi
fi
echo "Rented instance: $INSTANCE_ID"

# ---------------------------------------------------------------------------
# 5. Wait for SSH
# ---------------------------------------------------------------------------
echo ""
echo "=== Waiting for instance SSH (timeout: 2 min — Vast servers are unreliable, destroy fast) ==="
SSH_HOST=""
SSH_PORT=""
DEADLINE=$(( $(date +%s) + 120 ))

while [[ $(date +%s) -lt $DEADLINE ]]; do
  if $DRY_RUN; then
    SSH_HOST="dry-run-host"
    SSH_PORT="22"
    echo "[DRY-RUN] Skipping SSH wait."
    break
  fi

  INST_JSON=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || echo "{}")
  STATUS=$(echo "$INST_JSON" | jq -r '.actual_status // "unknown"')
  SSH_HOST=$(echo "$INST_JSON" | jq -r '.ssh_host // ""')
  SSH_PORT=$(echo "$INST_JSON" | jq -r '.ssh_port // ""')

  echo "  status=$STATUS  host=$SSH_HOST  port=$SSH_PORT"

  if [[ "$STATUS" == "running" && -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then
    if ssh -o StrictHostKeyChecking=no \
           -o ConnectTimeout=8 \
           -o ServerAliveInterval=15 \
           -o BatchMode=yes \
           -p "$SSH_PORT" \
           "root@$SSH_HOST" \
           "echo SSH_OK" 2>/dev/null | grep -q "SSH_OK"; then
      echo "SSH connection established."
      break
    fi
  fi

  if [[ $(date +%s) -ge $DEADLINE ]]; then
    echo "ERROR: SSH not available after 2 minutes — dead machine. Destroying and aborting." >&2
    vastai destroy instance "$INSTANCE_ID" --api-key "$VAST_API_KEY" || true
    INSTANCE_ID=""
    exit 1
  fi

  sleep 10
done

# ---------------------------------------------------------------------------
# 6. Deploy
# ---------------------------------------------------------------------------
echo ""
echo "=== Deploying benchmark environment ==="
if $DRY_RUN; then
  echo "[DRY-RUN] Would SSH to root@$SSH_HOST:$SSH_PORT and run setup commands."
else
  ssh -o StrictHostKeyChecking=no \
      -o ConnectTimeout=10 \
      -o ServerAliveInterval=30 \
      -p "$SSH_PORT" \
      "root@$SSH_HOST" << 'SETUP'
set -e

echo "--- Installing Python packages ---"
pip install vllm 2>/dev/null || true
pip install git+https://github.com/plumb-moe/plumb.git 2>/dev/null || true

# nvidia-nccl-cu13 (NCCL 2.28.9) requires CUDA 12.8+ runtime but torch ships cu12.
# Remove it and force cu12 (2.21.5) which matches torch's internal NCCL version.
pip uninstall -y nvidia-nccl-cu13 2>/dev/null || true
pip install 'nvidia-nccl-cu12==2.21.5' --force-reinstall --quiet 2>/dev/null || true

# Patch 1: vllm transformers_utils/tokenizer.py — transformers 5.9.0 removed
# all_special_tokens_extended from LlamaTokenizer (slow tokenizer); vllm 0.8.5
# accesses it unconditionally. Use getattr fallback instead.
python3 - << 'PATCH_EOF'
import os
path = '/opt/conda/lib/python3.11/site-packages/vllm/transformers_utils/tokenizer.py'
if os.path.exists(path):
    src = open(path).read()
    old = "tokenizer_all_special_tokens_extended = (\n        tokenizer.all_special_tokens_extended)"
    new = "tokenizer_all_special_tokens_extended = (\n        getattr(tokenizer, 'all_special_tokens_extended', []))"
    if old in src:
        open(path, 'w').write(src.replace(old, new, 1))
        print('Patch 1 applied: tokenizer.py')
    else:
        print('Patch 1 skipped (already applied or not needed)')
PATCH_EOF

# Patch 2: vllm model_executor/models/mixtral.py — AWQ config.json has
# "head_dim": null which overrides getattr fallback; use `or` to handle None.
python3 - << 'PATCH_EOF'
import os
path = '/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/mixtral.py'
if os.path.exists(path):
    src = open(path).read()
    old = 'self.head_dim = getattr(config, "head_dim",\n                               self.hidden_size // self.total_num_heads)'
    new = 'self.head_dim = (getattr(config, "head_dim", None)\n                          or self.hidden_size // self.total_num_heads)'
    if old in src:
        open(path, 'w').write(src.replace(old, new, 1))
        print('Patch 2 applied: mixtral.py')
    else:
        print('Patch 2 skipped (already applied or not needed)')
PATCH_EOF

# Patch 3: vllm config.py get_head_size — same null head_dim issue causes
# KV cache size calculation to return None.
python3 - << 'PATCH_EOF'
import os
path = '/opt/conda/lib/python3.11/site-packages/vllm/config.py'
if os.path.exists(path):
    src = open(path).read()
    old = '        if hasattr(self.hf_text_config, "head_dim"):\n            return self.hf_text_config.head_dim'
    new = '        if hasattr(self.hf_text_config, "head_dim") and self.hf_text_config.head_dim is not None:\n            return self.hf_text_config.head_dim'
    if old in src:
        open(path, 'w').write(src.replace(old, new, 1))
        print('Patch 3 applied: config.py')
    else:
        print('Patch 3 skipped (already applied or not needed)')
PATCH_EOF

# Install GCC — needed by torch inductor for AWQ Marlin kernel JIT compilation.
apt-get install -y gcc > /dev/null 2>&1 || true

echo "--- Downloading ShareGPT dataset ---"
python3 -c "
import urllib.request, os
dest = '/tmp/sharegpt.json'
if os.path.exists(dest):
    print('ShareGPT already present, skipping download.')
else:
    print('Downloading ShareGPT...')
    urllib.request.urlretrieve(
        'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json',
        dest
    )
    print('Done:', dest)
"

echo "--- Cloning plumb-oss repo ---"
if [ -d /opt/plumb-oss ]; then
  echo 'Repo already cloned, pulling latest...'
  git -C /opt/plumb-oss pull --ff-only || true
else
  git clone https://github.com/plumb-moe/plumb.git /opt/plumb-oss
fi

echo "Setup complete."
SETUP
fi

# ---------------------------------------------------------------------------
# 7. Run benchmark
# ---------------------------------------------------------------------------
echo ""
echo "=== Running benchmark (model=$MODEL, tp=$TP) ==="
if $DRY_RUN; then
  echo "[DRY-RUN] Would run sharegpt_moe_bench.py on remote host."
else
  ssh -o StrictHostKeyChecking=no \
      -o ConnectTimeout=10 \
      -o ServerAliveInterval=30 \
      -p "$SSH_PORT" \
      "root@$SSH_HOST" \
      MODEL="$MODEL" TOKENIZER="$TOKENIZER" TOKENIZER_MODE="$TOKENIZER_MODE" TP="$TP" bash << 'BENCH'
set -e
cd /opt/plumb-oss
echo "Starting benchmark: model=$MODEL tp=$TP tokenizer=${TOKENIZER:-<same as model>} tokenizer_mode=${TOKENIZER_MODE:-auto}"
mkdir -p /tmp/bench-results

EXTRA_ARGS=()
if [[ -n "$TOKENIZER" ]]; then
  EXTRA_ARGS+=(--tokenizer "$TOKENIZER")
fi
if [[ -n "$TOKENIZER_MODE" ]]; then
  EXTRA_ARGS+=(--tokenizer-mode "$TOKENIZER_MODE")
fi

PYTHONPATH=/opt/plumb-oss python bench/sharegpt_moe_bench.py \
  --model "$MODEL" \
  "${EXTRA_ARGS[@]}" \
  --tp "$TP" \
  --num-requests 500 \
  --concurrency 1,4,16,32 \
  --sharegpt-path /tmp/sharegpt.json \
  --output-dir /tmp/bench-results \
  --skip-phase2 \
  --trust-remote-code \
  --hetero-sim

echo "Benchmark done. Results in /tmp/bench-results/"
ls -lh /tmp/bench-results/
BENCH
fi

# ---------------------------------------------------------------------------
# 8. Retrieve results
# ---------------------------------------------------------------------------
echo ""
echo "=== Retrieving results ==="
LOCAL_RESULTS="bench/results/vast-$(date +%Y%m%d-%H%M%S)"
run mkdir -p "$LOCAL_RESULTS"

if $DRY_RUN; then
  echo "[DRY-RUN] Would scp root@$SSH_HOST:/tmp/bench-results/* -> $LOCAL_RESULTS/"
else
  scp -o StrictHostKeyChecking=no \
      -o ConnectTimeout=10 \
      -P "$SSH_PORT" \
      -r \
      "root@$SSH_HOST:/tmp/bench-results/*" \
      "$LOCAL_RESULTS/"
  echo "Results saved to $LOCAL_RESULTS"
fi

# ---------------------------------------------------------------------------
# 9. Generate charts
# ---------------------------------------------------------------------------
echo ""
echo "=== Generating charts ==="
run python bench/charts/plot_results.py \
  --results-dir "$LOCAL_RESULTS" \
  --output-dir "$LOCAL_RESULTS/charts"
echo "Charts saved to $LOCAL_RESULTS/charts/"

# ---------------------------------------------------------------------------
# 10. Destroy instance
# ---------------------------------------------------------------------------
echo ""
echo "=== Destroying instance $INSTANCE_ID ==="
run vastai destroy instance "$INSTANCE_ID" --api-key "$VAST_API_KEY"
echo "Instance $INSTANCE_ID destroyed."
# Clear INSTANCE_ID so the EXIT trap does not double-destroy
INSTANCE_ID=""

# ---------------------------------------------------------------------------
# 11. Print summary
# ---------------------------------------------------------------------------
echo ""
echo "=== BENCHMARK RESULTS ==="
SUMMARY="$LOCAL_RESULTS/summary.json"
if [[ -f "$SUMMARY" ]]; then
  jq -r '"Model:                " + .model'                                    "$SUMMARY" || true
  jq -r '"Phase 1 throughput:   " + (.phase1_throughput_rps | tostring) + " req/s"' "$SUMMARY" || true
  jq -r '"Cross-GPU dispatch:   " + (.cross_gpu_dispatch_rate | tostring)'     "$SUMMARY" || true
  echo "Charts: $LOCAL_RESULTS/charts/"
else
  echo "No summary.json found in $LOCAL_RESULTS — check raw result files."
  ls -lh "$LOCAL_RESULTS/" 2>/dev/null || true
fi

echo ""
echo "Run complete."
