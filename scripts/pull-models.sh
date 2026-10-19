#!/usr/bin/env bash
# Pull open-weight GGUF models straight from Hugging Face into Ollama.
# Usage: ./scripts/pull-models.sh [hf.co/user/repo:QUANT ...]
# With no args, pulls CHAT_MODEL from .env.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
  set -- "$(grep -E '^CHAT_MODEL=' .env | cut -d= -f2-)"
fi

for model in "$@"; do
  echo "Pulling $model"
  docker compose exec ollama ollama pull "$model"
done
docker compose exec ollama ollama list
