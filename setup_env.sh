#!/usr/bin/env bash
# Creates a uv-managed venv at /data/fitllm-venv and installs all dependencies.
# Run from the FitLLM project root:  bash setup_env.sh

set -e

VENV_DIR="/data/fitllm-venv"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

echo "==> Creating venv at $VENV_DIR"
uv venv "$VENV_DIR" --python 3.11

echo "==> Activating venv"
source "$VENV_DIR/bin/activate"

echo "==> Installing PyTorch (CUDA 12.1)"
uv pip install torch==2.3.0 torchvision --index-url "$TORCH_INDEX"

echo "==> Installing core dependencies"
uv pip install -r requirements.txt

echo "==> Installing FitLLM package (editable)"
uv pip install -e .

echo ""
echo "===================================================="
echo "  Core install complete."
echo "  To install optional acceleration libraries run:"
echo "    source $VENV_DIR/bin/activate"
echo "    uv pip install -r requirements-acceleration.txt --no-build-isolation"
echo "===================================================="
echo ""
echo "  To activate this environment in future sessions:"
echo "    source $VENV_DIR/bin/activate"
echo ""
