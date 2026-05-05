#!/usr/bin/env bash
# Set up the fulcrum environment.
#
# - Installs uv (fast Python package manager) if missing
# - Creates .venv with Python 3.10+
# - Installs fulcrum (this package) editable, plus Murmura from GitHub
# - Installs FLamby from source (it is not on PyPI)
#
# Usage:  bash scripts/setup_env.sh

set -euo pipefail

# Resolve repo root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Setting up fulcrum at $REPO_ROOT"

# Warn loudly if a conda env is active — uv venv + conda often layer in
# confusing ways (PATH order, shebang resolution). Run `conda deactivate`
# first to keep the install isolated to the uv venv.
if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    echo "WARNING: conda env '${CONDA_DEFAULT_ENV}' is active." >&2
    echo "         Layering a uv venv on top can confuse PATH and pytest invocation." >&2
    echo "         Recommend: 'conda deactivate' before re-running this script." >&2
    echo "         Continuing anyway in 5 seconds..." >&2
    sleep 5
fi

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

# Create venv and install fulcrum (which pulls in Murmura)
echo "==> Creating .venv"
uv venv --python 3.10
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing fulcrum + dependencies (includes Murmura from GitHub + pytest)"
uv pip install -e ".[dev]"

# FLamby is not on PyPI — install from source for the medical datasets
echo "==> Installing FLamby from source (Fed-ISIC2019, Fed-Heart-Disease)"
FLAMBY_DIR="$REPO_ROOT/.flamby_src"
if [ ! -d "$FLAMBY_DIR" ]; then
    git clone --depth 1 https://github.com/owkin/FLamby.git "$FLAMBY_DIR"
fi
uv pip install -e "$FLAMBY_DIR"

# Optional dataset extras (Fed-Heart-Disease has a pip extra; Fed-ISIC2019 needs manual download — see download_data.py)
uv pip install -e "$FLAMBY_DIR[heart_disease]" || true

echo
echo "==> Done. Activate the venv with:  source .venv/bin/activate"
echo "==> Then download datasets:        python scripts/download_data.py"
