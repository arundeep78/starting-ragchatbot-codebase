#!/bin/bash
set -e

echo "🔍 Running Linting Checks..."
echo "============================="

# Change to project root
cd "$(dirname "$0")/.."

# Install dev dependencies if needed
uv sync --group dev

echo "Running flake8 linter..."
uv run flake8 backend/ main.py

# echo "Running mypy type checker..."
# uv run mypy backend/ main.py

echo "✅ Linting checks complete!"