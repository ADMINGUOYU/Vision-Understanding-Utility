#!/usr/bin/env bash
set -euo pipefail

# This script downloads the Qwen3-VL-8B-Instruct model
# and sets up the environment packages.

# --------------- Configuration ---------------
# Or specify with the FIRST argument
# We'll install the packages in this Conda environment
CONDA_ENV_NAME="${1:-VisPROC}"
# Or specify with the SECOND argument
# We'll download the model to this directory
MODEL_DIR="${2:-./data/Qwen/Qwen3-VL-8B-Instruct}"
# Model ID on Hugging Face Hub
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
# --------------- Configuration ---------------

# Export huggingface cache directory to avoid permission issues
# HF_HOME and TRANSFORMERS_CACHE
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/huggingface/transformers"
# TORCH_HOME for PyTorch models
export TORCH_HOME="./cache/torch"

# Make sure conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo -e  "\033[1;31m[ERROR]\033[0m conda not found. Please install Conda first."
  exit 1
fi

# Activate the Conda environment
echo -e "\033[1;34m[INFO]\033[0m Activating Conda environment: ${CONDA_ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

# Install required packages
echo -e "\033[1;34m[INFO]\033[0m Installing required packages in Conda environment: ${CONDA_ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install \
  "transformers>=4.51.0" \
  "accelerate" \
  "sentence-transformers" \
  "qwen-vl-utils" \
  "pillow" \
  "tqdm"

# Download the model from Hugging Face Hub
echo -e "\033[1;34m[INFO]\033[0m Creating model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"
echo -e "\033[1;34m[INFO]\033[0m Downloading model: ${MODEL_ID} to ${MODEL_DIR}"
# Use data_process/setup_captioning/hf_downloader.py to download
python -m utility.hf_downloader \
  --repo_id "${MODEL_ID}" \
  --target_path "${MODEL_DIR}" \
  --repo_type "model"

# DONE
echo -e "\033[1;32m[OK]\033[0m Qwen3-VL-8B-Instruct setup completed."
echo -e "\033[1;32m[OK]\033[0m Model downloaded to: ${MODEL_DIR}"