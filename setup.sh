#!/bin/bash

# set -euo pipefail:
# -e: Exit immediately if a command exits with a non-zero status.
# -u: Treat unset variables as an error and exit immediately.
# -o pipefail: Return the exit status of the last command in the pipeline that failed
#              (i.e., the first non-zero exit status) instead of the exit status of the last command.
set -euo pipefail

# This script sets up the environment AND
# downloads the selected QWEN model and dependencies.
# Available models:
# - Qwen3-VL-8B-Instruct

# ---------------- Configuration ----------------
MODEL_NAME="Qwen3-VL-8B-Instruct"
MODEL_DIR="./data/Qwen/$MODEL_NAME"
# ------------- Configuration ENDs --------------

# Fetch the model installation script from ./setup/...
# script naming: setup_{MODEL_NAME}.sh
SETUP_SCRIPT="./setup/setup_${MODEL_NAME}.sh"
# Check if the setup script exists
if [[ ! -f "$SETUP_SCRIPT" ]]; then
    echo -e  "\033[1;31m[ERROR]\033[0m Setup script for model '$MODEL_NAME' not found at '$SETUP_SCRIPT'."
    exit 1
fi

# Fetch ./environment.yml
ENV_FILE="./environment.yml"
# Get environment name from the .yml file
ENV_NAME=$(grep -E '^name:' "$ENV_FILE" | awk '{print $2}')
if [[ -z "$ENV_NAME" ]]; then
    echo -e  "\033[1;31m[ERROR]\033[0m Could not find environment name in '$ENV_FILE'."
    exit 1
fi

# Make sure conda is available
if ! command -v conda &> /dev/null; then
    echo -e  "\033[1;31m[ERROR]\033[0m Conda command not found. Please ensure Conda is installed and added to your PATH."
    exit 1
fi

# Check if the environment already exists
# if ask user if they want to overwrite the existing environment
if conda env list | grep -q "$ENV_NAME"; then
    echo -e  "\033[1;33m[WARNING]\033[0m Conda environment '$ENV_NAME' already exists."
    echo -e  "\033[1;33m[WARNING]\033[0m Do you want to overwrite it? (y/n): "
    read -r answer
    if [[ "$answer" != "y" ]]; then
        # answer == "n"
        # Prompt user and exit without making any changes
        echo -e  "\033[1;32m[INFO]\033[0m Abort. No changes have been made to the existing Conda environment '$ENV_NAME'."
        exit 0
    else
        # answer == "y"
        # Update the existing environment with the new dependencies
        echo -e  "\033[1;32m[INFO]\033[0m Updating existing Conda environment '$ENV_NAME' with dependencies from '$ENV_FILE'"
        conda env update -n "$ENV_NAME" -f "$ENV_FILE"
    fi
else
    echo -e  "\033[1;32m[INFO]\033[0m Creating Conda environment '$ENV_NAME' from '$ENV_FILE'"
    conda env create -f "$ENV_FILE"
fi

# Activate the environment
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Run the model-specific setup script
echo -e  "\033[1;32m[INFO]\033[0m Running setup script for model '$MODEL_NAME' at '$SETUP_SCRIPT'"
# Setup script has TWO arguments (env name, model dir)
bash "$SETUP_SCRIPT" "$ENV_NAME" "$MODEL_DIR"

# DONE - prompt user how to activate and remove the environment
echo -e  "\033[1;32m[OK]\033[0m Setup for model '$MODEL_NAME' completed successfully."
echo -e  "\033[1;32m[INFO]\033[0m To activate the Conda environment, run: \033[1;34mconda activate $ENV_NAME\033[0m"
echo -e  "\033[1;32m[INFO]\033[0m To remove the Conda environment, run: \033[1;34mconda env remove -n $ENV_NAME\033[0m"
