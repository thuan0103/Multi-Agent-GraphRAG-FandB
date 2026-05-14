#!/usr/bin/env bash
# B2.1 — Launch Dual-Model SGLang on 12GB VRAM (RTX 3060)
# Folder: services/llm_serving/launch_sglang.sh
#
# Router:    0.5B–1.5B AWQ  → ~1.1 GB VRAM  (10%)
# Generator: 7B AWQ         → ~4.8 GB VRAM  (40%)
# KV Cache:  remainder      → ~6.1 GB VRAM  (50%)

set -e

ROUTER_MODEL=${ROUTER_MODEL:-"Qwen/Qwen2.5-1.5B-Instruct-AWQ"}
GENERATOR_MODEL=${GENERATOR_MODEL:-"Qwen/Qwen2.5-7B-Instruct-AWQ"}

echo "=== Launching Router Model ==="
python -m sglang.launch_server \
  --model-path "$ROUTER_MODEL" \
  --port 8080 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.09 \
  --context-length 512 \
  --schedule-policy lpm \
  --chunked-prefill-size 512 \
  --max-running-requests 32 \
  --quantization awq \
  --dtype half \
  &

ROUTER_PID=$!
echo "Router PID: $ROUTER_PID"

# Wait for router to be ready
until curl -sf http://localhost:8080/health > /dev/null; do
  echo "Waiting for router..."
  sleep 2
done
echo "Router ready."

echo "=== Launching Generator Model ==="
python -m sglang.launch_server \
  --model-path "$GENERATOR_MODEL" \
  --port 8081 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.78 \
  --context-length 4096 \
  --schedule-policy lpm \
  --chunked-prefill-size 2048 \
  --max-running-requests 8 \
  --quantization awq \
  --dtype half \
  &

GENERATOR_PID=$!
echo "Generator PID: $GENERATOR_PID"

until curl -sf http://localhost:8081/health > /dev/null; do
  echo "Waiting for generator..."
  sleep 3
done
echo "Generator ready."

echo "=== Both models ready ==="
echo "Router:    http://localhost:8080"
echo "Generator: http://localhost:8081"

wait $ROUTER_PID $GENERATOR_PID
