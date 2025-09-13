#!/bin/bash
set -e

echo "🧪 Running Tests..."
echo "==================="

# Change to project root
cd "$(dirname "$0")/.."

# Install dependencies
uv sync

# Check if tests directory exists
if [ -d "backend/tests" ]; then
    echo "Running pytest..."
    uv run pytest backend/tests/ -v --tb=short
    echo "✅ All tests passed!"
else
    echo "ℹ️  No tests directory found."
fi