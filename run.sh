#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 &
npm run dev -- --host 127.0.0.1
