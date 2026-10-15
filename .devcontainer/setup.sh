#!/usr/bin/env bash
set -euo pipefail

pip install --quiet --user uv==0.10.0
export PATH="$HOME/.local/bin:$PATH"

uv sync --frozen --no-dev
(cd web && npm ci --no-audit --no-fund && npm run build)

uv run --no-dev python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-base-en-v1.5', cache_dir='.cache/fastembed')"

echo "SARAL is installed. It starts automatically; see docs/DEPLOYMENT.md."
