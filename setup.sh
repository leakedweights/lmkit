#!/usr/bin/env bash
set -e

# Check and install astral-sh UV if it is not already installed.
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing astral-sh UV..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
else
  echo "astral-sh UV is already installed."
fi

# Create the virtual environment using uv if the "venv" directory does not exist.
if [ ! -d "venv" ]; then
  echo "Creating virtual environment with uv..."
  uv venv
else
  echo "Virtual environment already exists."
fi

# Activate the virtual environment.
echo "Activating the virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# Run uv sync with CUDA extras if nvcc is available; otherwise, run uv sync normally.
if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc detected. Running 'uv sync --extra cuda'..."
  uv sync --extra cuda
else
  echo "nvcc not found. Running 'uv sync'..."
  uv sync
fi

# Install Visual Studio Code extensions.
echo "Installing Visual Studio Code extensions..."
code --install-extension charliermarsh.ruff
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter

echo "Setup complete."
