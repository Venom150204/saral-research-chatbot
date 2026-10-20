#!/usr/bin/env bash
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

# SSH sessions (used by the keep-codespace-up workflow) do not inherit the
# Codespaces secrets, so read them from the container's own environment.
if [[ -z "${GEMINI_API_KEY:-}" ]] && sudo -n true 2>/dev/null; then
  while IFS= read -r -d '' kv; do
    case "$kv" in
      GEMINI_API_KEY=*|SARAL_DATABASE_URL=*|DATABASE_URL_UNPOOLED=*|DATABASE_URL=*) export "$kv" ;;
    esac
  done < <(sudo cat /proc/1/environ 2>/dev/null)
fi

if [[ -z "${GEMINI_API_KEY:-}" || -z "${SARAL_DATABASE_URL:-}${DATABASE_URL_UNPOOLED:-}${DATABASE_URL:-}" ]]; then
  echo "SARAL not started: add the GEMINI_API_KEY and SARAL_DATABASE_URL Codespaces secrets"
  echo "(github.com > Settings > Codespaces > Secrets), then rebuild or restart the codespace."
  exit 0
fi

if curl -fs -m 5 -o /dev/null http://localhost:8000/health; then
  echo "SARAL is already running on port 8000 (log: /tmp/saral.log)."
  exit 0
fi

pkill -f "uvicorn backend.app.api:app" 2>/dev/null || true
setsid nohup uv run --no-dev uvicorn backend.app.api:app --host 0.0.0.0 --port 8000 \
  --proxy-headers --forwarded-allow-ips='*' > /tmp/saral.log 2>&1 < /dev/null &
disown || true
echo "SARAL is starting on port 8000 (log: /tmp/saral.log)."
