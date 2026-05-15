#!/usr/bin/env bash

set -euo pipefail

# ---------------- Configuration ----------------
# Model paths
MODEL_PATH="data/Qwen/Qwen3-VL-8B-Instruct"
SECONDARY_MODEL_PATH="$MODEL_PATH"
# Device to use
DEVICE_MAP="cuda:0"
# Web server port
WEB_PORT="6841"
# How MANY completed jobs to keep
COMPLETED_JOB_RETENTION="5"

TORCH_DTYPE="${TORCH_DTYPE:-auto}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
CUDA_VISIBLE_DEVICES=${DEVICE_MAP#cuda:}
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
# ------------- Configuration ENDs --------------

# HF_HOME and TRANSFORMERS_CACHE
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/huggingface/transformers"
# Export TORCH_HOME
export TORCH_HOME="./cache/torch"

# Export environment variables for the server
export MODEL_PATH
export SECONDARY_MODEL_PATH
export CUDA_VISIBLE_DEVICES
export DEVICE_MAP
export TORCH_DTYPE
export WEB_HOST
export WEB_PORT
export MAX_NEW_TOKENS

# Start the web server
python -m app.server.web
