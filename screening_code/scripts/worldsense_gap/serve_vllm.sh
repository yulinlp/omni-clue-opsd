#!/usr/bin/env bash
# Start a Qwen2.5-Omni-7B vLLM OpenAI service for one gap view.
# Usage: serve_vllm.sh <gpu_index> <port> <log_file>
set -uo pipefail
GPU="${1:?usage: serve_vllm.sh <gpu_index> <port> <log_file>}"
PORT="${2:?usage: serve_vllm.sh <gpu_index> <port> <log_file>}"
LOG="${3:?usage: serve_vllm.sh <gpu_index> <port> <log_file>}"
MODEL=/share/home/ylhu/models/Qwen2.5-Omni-7B
VLLM=/share/home/ylhu/.conda/envs/vllm/bin/vllm

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_LOGGING_LEVEL=INFO
mkdir -p "$(dirname "$LOG")"
setsid nohup "$VLLM" serve "$MODEL" \
  --served-model-name Qwen2.5-Omni-7B \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"video": 8, "audio": 8}' \
  --disable-log-requests \
  >> "$LOG" 2>&1 < /dev/null &
echo "vllm pid=$! gpu=$GPU port=$PORT log=$LOG"
