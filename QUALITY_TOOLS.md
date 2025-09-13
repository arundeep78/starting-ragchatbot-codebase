# Code Quality Tools

This project includes a comprehensive code quality toolchain to ensure consistent, maintainable code.

## Tools Included

- **Black**: Automatic code formatting (88 character line length)
- **isort**: Import sorting that's compatible with Black
- **flake8**: Linting for code style and basic errors
- **mypy**: Static type checking (configured but currently disabled)
- **pytest**: Unit testing framework
- **pre-commit**: Git hooks for automated quality checks

## Quick Start

### Install Development Dependencies
```bash
uv sync --group dev
```

### Available Scripts

#### Format Code
```bash
./scripts/format.sh
```
Automatically formats all Python code with Black and sorts imports with isort.

#### Run All Quality Checks
```bash
./scripts/quality.sh
```
Runs the complete quality pipeline:
- ✅ Black format check
- ✅ Import sort check
- ✅ Flake8 linting
- ✅ pytest tests

#### Run Individual Checks
```bash
./scripts/lint.sh    # Flake8 linting only
./scripts/test.sh    # Tests only
```

## Configuration

### Black Configuration
- Line length: 88 characters
- Target Python version: 3.12
- Excludes: git files, cache dirs, build dirs, chroma_db

### isort Configuration
- Black-compatible profile
- Multi-line output style 3
- Trailing commas enforced

### flake8 Configuration
- Max line length: 88 characters
- Relaxed rules for existing codebase
- Excludes test files from some strict checks

### pytest Configuration
- Runs all tests in `backend/tests/`
- Verbose output with short traceback format

## Pre-commit Hooks (Optional)

Set up automatic quality checks on every commit:

```bash
uv run pre-commit install
```

This will automatically run formatting and linting checks before each commit.

## Development Workflow

1. **Before starting work**: `uv sync --group dev`
2. **While developing**: Use `./scripts/format.sh` regularly
3. **Before committing**: Run `./scripts/quality.sh` to ensure all checks pass
4. **Optional**: Set up pre-commit hooks for automatic checks

## Notes

- MyPy type checking is configured but currently disabled due to existing type issues
- Flake8 has relaxed rules to accommodate the current codebase
- All 83 tests currently pass
- The quality script provides colored output and helpful fix suggestions

## Future Improvements

- Gradually enable stricter mypy settings as type annotations are added
- Remove flake8 exemptions as code quality improves
- Add additional linting tools (bandit for security, etc.)
- Integrate quality checks into CI/CD pipeline