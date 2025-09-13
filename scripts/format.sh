#!/bin/bash
set -e

echo "🎨 Formatting Code with Black and isort..."
echo "=========================================="

# Change to project root
cd "$(dirname "$0")/.."

# Install dev dependencies if needed
uv sync --group dev

echo "Running Black formatter..."
uv run black backend/ main.py

echo "Running isort import sorter..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"