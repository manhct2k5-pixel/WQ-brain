#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${PORT:=8000}"
echo "Starting Agentic AI-BI MVP on http://127.0.0.1:${PORT}"
python3 server.py
