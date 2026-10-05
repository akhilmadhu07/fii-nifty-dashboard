#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
streamlit run src/dashboard/app.py --server.headless true
