#!/usr/bin/env python3
"""
Test runner script for the RAG system backend
Provides convenient commands for running different test suites
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the result"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=Path(__file__).parent)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        return False


def main():
    """Main test runner"""
    if len(sys.argv) < 2:
        print("📋 RAG System Test Runner")
        print("=" * 40)
        print("Available commands:")
        print("  all          - Run all tests")
        print("  unit         - Run only unit tests")
        print("  api          - Run only API tests")
        print("  integration  - Run only integration tests")
        print("  slow         - Run all tests including slow ones")
        print("  coverage     - Run tests with coverage report")
        print("  lint         - Run linting checks")
        print("  quick        - Run fast tests only (exclude slow)")
        print("\nExample: python run_tests.py all")
        sys.exit(1)

    command = sys.argv[1].lower()

    # Change to backend directory
    os.chdir(Path(__file__).parent)

    success_count = 0
    total_count = 0

    if command == "all":
        total_count = 1
        if run_command("python -m pytest tests/ -v", "Running all tests"):
            success_count += 1

    elif command == "unit":
        total_count = 1
        if run_command("python -m pytest tests/ -v -m unit", "Running unit tests"):
            success_count += 1

    elif command == "api":
        total_count = 1
        if run_command("python -m pytest tests/test_api_endpoints.py -v", "Running API tests"):
            success_count += 1

    elif command == "integration":
        total_count = 1
        if run_command("python -m pytest tests/ -v -m integration", "Running integration tests"):
            success_count += 1

    elif command == "slow":
        total_count = 1
        if run_command("python -m pytest tests/ -v --tb=short", "Running all tests including slow ones"):
            success_count += 1

    elif command == "coverage":
        total_count = 2
        if run_command("python -m pytest tests/ --cov=. --cov-report=html --cov-report=term", "Running tests with coverage"):
            success_count += 1
        print("\n📊 Coverage report generated in htmlcov/index.html")

    elif command == "lint":
        print("⚠️  Linting not configured yet. Use 'uv run ruff check .' if ruff is available")

    elif command == "quick":
        total_count = 1
        if run_command('python -m pytest tests/ -v -m "not slow"', "Running quick tests only"):
            success_count += 1

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    if success_count == total_count:
        print(f"✅ All test suites passed ({success_count}/{total_count})")
        sys.exit(0)
    else:
        print(f"❌ Some test suites failed ({success_count}/{total_count})")
        sys.exit(1)


if __name__ == "__main__":
    main()