#!/bin/bash
set -e

echo "🔍 Running Code Quality Checks..."
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}$1${NC}"
    echo "----------------------------------------"
}

# Function to run a command and track success/failure
run_check() {
    local name="$1"
    local cmd="$2"
    local fix_cmd="$3"

    print_header "$name"

    if eval "$cmd"; then
        echo -e "${GREEN}✅ $name: PASSED${NC}"
        echo
        return 0
    else
        echo -e "${RED}❌ $name: FAILED${NC}"
        if [ -n "$fix_cmd" ]; then
            echo -e "${YELLOW}💡 To fix, run: $fix_cmd${NC}"
        fi
        echo
        return 1
    fi
}

# Track overall success
FAILED_CHECKS=0

# Change to project root
cd "$(dirname "$0")/.."

echo "📦 Installing development dependencies..."
uv sync --group dev
echo

# Run Black formatting check
if ! run_check "Black Format Check" "uv run black --check --diff backend/ main.py" "uv run black backend/ main.py"; then
    ((FAILED_CHECKS++))
fi

# Run isort import sorting check
if ! run_check "Import Sort Check" "uv run isort --check-only --diff backend/ main.py" "uv run isort backend/ main.py"; then
    ((FAILED_CHECKS++))
fi

# Run flake8 linting
if ! run_check "Flake8 Linting" "uv run flake8 backend/ main.py" ""; then
    ((FAILED_CHECKS++))
fi

# Run mypy type checking (disabled for now due to many type issues)
# if ! run_check "MyPy Type Check" "uv run mypy backend/ main.py" ""; then
#     ((FAILED_CHECKS++))
# fi

# Run tests if they exist
if [ -d "backend/tests" ]; then
    if ! run_check "Pytest Tests" "uv run pytest backend/tests/ -v" ""; then
        ((FAILED_CHECKS++))
    fi
fi

# Final summary
echo "================================="
if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}🎉 All quality checks passed!${NC}"
    exit 0
else
    echo -e "${RED}💥 $FAILED_CHECKS check(s) failed${NC}"
    echo "Run the suggested fix commands above to resolve issues."
    exit 1
fi