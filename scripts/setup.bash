#!/usr/bin/env bash

mkdir -p output

# Setup environment
if ! command -v uv >/dev/null 2>&1; then
    warn "uv is not installed. Install it with:"
    warn "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    warn "then re-run this script."
    exit 1
fi

# pyproject.toml pins torch to the CUDA 12.6 wheels, which exist only for Linux x86_64 and Windows
if [[ "$(uname -s)" != "Linux" ]]; then
    warn "torch is pinned to CUDA 12.6 wheels, which uv cannot install on $(uname -s)."
    warn "Skipping 'uv sync'; set up the environment on a Linux machine with a GPU."
else
    info "installing the Python environment with uv (Python $(cat .python-version))"
    uv sync
    uv run python -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"
fi
