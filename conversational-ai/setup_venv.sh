#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
echo "Virtual environment created in .venv"
echo "To activate: source .venv/bin/activate"
