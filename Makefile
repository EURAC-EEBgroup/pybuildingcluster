# Makefile for PyBuildingCluster Library
# 
# This Makefile provides convenient commands for development, testing, and deployment tasks.

.PHONY: help install install-dev test test-all lint format clean build docs serve-docs publish-test publish

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON = python
PIP = pip
TOX = tox
PYTEST = pytest
BLACK = black
FLAKE8 = flake8
MYPY = mypy
PACKAGE_NAME = geoclustering
TEST_DIR = tests
DOCS_DIR = docs

# Colors for output
BLUE = \033[36m
GREEN = \033[32m
YELLOW = \033[33m
RED = \033[31m
NC = \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)PyBuildingCluster - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make install-dev    # Install in development mode"
	@echo "  make test          # Run basic tests"
	@echo "  make test-all      # Run all tests with tox"
	@echo "  make lint          # Run all linting tools"
	@echo "  make format        # Auto-format code"
	@echo "  make docs          # Build documentation"

install: ## Install package in production mode
	@echo "$(BLUE)Installing package...$(NC)"
	$(PIP) install -e .

install-dev: ## Install package in development mode with all dependencies
	@echo "$(BLUE)Installing package in development mode...$(NC)"
	$(PIP) install -e ".[dev,docs,interactive]"
	@echo "$(GREEN)Development installation complete!$(NC)"

install-test: ## Install test dependencies only
	@echo "$(BLUE)Installing test dependencies...$(NC)"
	$(PIP) install -e ".[dev]"

test: ## Run basic test suite
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --cov=$(PACKAGE_NAME) --cov-report=term-missing

test-fast: ## Run tests with minimal output
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -x -q

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -m "unit" -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -m "integration" -v

test-slow: ## Run all tests including slow ones
	@echo "$(BLUE)Running all tests including slow ones...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --cov=$(PACKAGE_NAME)

test-all: ## Run tests across all Python versions using tox
	@echo "$(BLUE)Running tests across all Python versions...$(NC)"
	$(TOX)

test-coverage: ## Run tests with detailed coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(TOX) -e coverage
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-performance: ## Run performance benchmarks
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(TOX) -e performance

lint: ## Run all linting tools
	@echo "$(BLUE)Running linting tools...$(NC)"
	$(TOX) -e lint

lint-flake8: ## Run flake8 linting
	@echo "$(BLUE)Running flake8...$(NC)"
	$(FLAKE8) $(PACKAGE_NAME) $(TEST_DIR)

lint-mypy: ## Run mypy type checking
	@echo "$(BLUE)Running mypy...$(NC)"
	$(MYPY) $(PACKAGE_NAME)

lint-black: ## Check code formatting with black
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(BLACK) --check --diff $(PACKAGE_NAME) $(TEST_DIR)

format: ## Auto-format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(TOX) -e format
	@echo "$(GREEN)Code formatting complete!$(NC)"

format-black: ## Format code with black only
	@echo "$(BLUE)Formatting with black...$(NC)"
	$(BLACK) $(PACKAGE_NAME) $(TEST_DIR)

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	$(TOX) -e security

clean: ## Clean up build artifacts and cache files
	@echo "$(BLUE)Cleaning up...$(NC)"
	$(TOX) -e clean
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .tox
	rm -rf htmlcov
	rm -rf .coverage
	@echo "$(GREEN)Cleanup complete!$(NC)"

build: ## Build package for distribution
	@echo "$(BLUE)Building package...$(NC)"
	$(TOX) -e build
	@echo "$(GREEN)Package built successfully! Check dist/ directory$(NC)"

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	$(TOX) -e docs
	@echo "$(GREEN)Documentation built! Open docs/_build/html/index.html$(NC)"

docs-serve: ## Build and serve documentation with auto-reload
	@echo "$(BLUE)Starting documentation server...$(NC)"
	@echo "$(YELLOW)Documentation will be available at http://localhost:8000$(NC)"
	$(TOX) -e docs-serve

docs-clean: ## Clean documentation build files
	@echo "$(BLUE)Cleaning documentation...$(NC)"
	rm -rf $(DOCS_DIR)/_build
	@echo "$(GREEN)Documentation cleaned!$(NC)"

check: ## Run all checks (lint, test, security)
	@echo "$(BLUE)Running all checks...$(NC)"
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) security
	@echo "$(GREEN)All checks passed!$(NC)"

dev-setup: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install tox pre-commit
	$(MAKE) install-dev
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

publish-test: ## Publish to TestPyPI
	@echo "$(BLUE)Publishing to TestPyPI...$(NC)"
	$(MAKE) clean
	$(MAKE) build
	$(TOX) -e publish-test
	@echo "$(GREEN)Published to TestPyPI!$(NC)"

publish: ## Publish to PyPI (production)
	@echo "$(YELLOW)WARNING: This will publish to production PyPI!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BLUE)Publishing to PyPI...$(NC)"; \
		$(MAKE) clean; \
		$(MAKE) build; \
		$(TOX) -e publish; \
		echo "$(GREEN)Published to PyPI!$(NC)"; \
	else \
		echo "$(YELLOW)Publication cancelled.$(NC)"; \
	fi

release: ## Create a new release (clean, test, build, tag)
	@echo "$(BLUE)Creating release...$(NC)"
	$(MAKE) clean
	$(MAKE) check
	$(MAKE) build
	@echo "$(GREEN)Release ready! Don't forget to:$(NC)"
	@echo "  1. Update version in pyproject.toml"
	@echo "  2. Update CHANGELOG.md"
	@echo "  3. Create git tag: git tag v1.0.1"
	@echo "  4. Push tag: git push origin v1.0.1"
	@echo "  5. Run 'make publish' to upload to PyPI"

env-info: ## Display environment information
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Package name: $(PACKAGE_NAME)"
	@echo "Test directory: $(TEST_DIR)"
	@if command -v $(TOX) >/dev/null 2>&1; then \
		echo "Tox version: $$($(TOX) --version)"; \
	else \
		echo "$(YELLOW)Tox not installed$(NC)"; \
	fi

install-hooks: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

run-hooks: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

example: ## Run example analysis (requires sample data)
	@echo "$(BLUE)Running example analysis...$(NC)"
	$(PYTHON) -c "from geoclustering import GeoClusteringAnalyzer; print('Library imported successfully!')"
	@echo "$(GREEN)Example completed! Check the library documentation for detailed usage.$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTEST) $(TEST_DIR) --benchmark-only --benchmark-sort=mean
	@echo "$(GREEN)Benchmarks completed!$(NC)"

profile: ## Profile the code (requires line_profiler)
	@echo "$(BLUE)Profiling code...$(NC)"
	@if command -v kernprof >/dev/null 2>&1; then \
		kernprof -l -v examples/profile_example.py; \
	else \
		echo "$(YELLOW)line_profiler not installed. Install with: pip install line_profiler$(NC)"; \
	fi

deps-update: ## Update dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -e ".[dev,docs,interactive]"
	@echo "$(GREEN)Dependencies updated!$(NC)"

deps-check: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	$(PIP) list --outdated

# Platform-specific commands
ifeq ($(OS),Windows_NT)
    # Windows-specific commands
    RM = del /Q
    RMDIR = rmdir /S /Q
    MKDIR = mkdir
    OPEN = start
else
    # Unix-like systems (Linux, macOS)
    RM = rm -f
    RMDIR = rm -rf
    MKDIR = mkdir -p
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        OPEN = open
    else
        OPEN = xdg-open
    endif
endif

open-docs: ## Open documentation in browser
	@echo "$(BLUE)Opening documentation...$(NC)"
	$(OPEN) docs/_build/html/index.html

open-coverage: ## Open coverage report in browser
	@echo "$(BLUE)Opening coverage report...$(NC)"
	$(OPEN) htmlcov/index.html

docker-build: ## Build Docker image for development
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t geoclustering-dev .
	@echo "$(GREEN)Docker image built!$(NC)"

docker-test: ## Run tests in Docker container
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	docker run --rm -v $(PWD):/app geoclustering-dev make test

docker-shell: ## Start interactive shell in Docker container
	@echo "$(BLUE)Starting Docker shell...$(NC)"
	docker run --rm -it -v $(PWD):/app geoclustering-dev /bin/bash

# Convenience aliases
t: test ## Alias for test
ta: test-all ## Alias for test-all
l: lint ## Alias for lint
f: format ## Alias for format
c: clean ## Alias for clean
b: build ## Alias for build
d: docs ## Alias for docs
i: install-dev ## Alias for install-dev

# Advanced testing commands
test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)Starting test watch mode...$(NC)"
	@if command -v ptw >/dev/null 2>&1; then \
		ptw $(TEST_DIR) -- -v; \
	else \
		echo "$(YELLOW)pytest-watch not installed. Install with: pip install pytest-watch$(NC)"; \
	fi

test-debug: ## Run tests with debugging enabled
	@echo "$(BLUE)Running tests with debugging...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -s --pdb

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -n auto

test-random: ## Run tests in random order
	@echo "$(BLUE)Running tests in random order...$(NC)"
	$(PYTEST) $(TEST_DIR) --random-order

# Documentation helpers
docs-api: ## Generate API documentation automatically
	@echo "$(BLUE)Generating API documentation...$(NC)"
	sphinx-apidoc -o $(DOCS_DIR)/api $(PACKAGE_NAME) --force --module-first
	@echo "$(GREEN)API documentation generated!$(NC)"

docs-linkcheck: ## Check for broken links in documentation
	@echo "$(BLUE)Checking documentation links...$(NC)"
	cd $(DOCS_DIR) && sphinx-build -b linkcheck . _build/linkcheck

# Quality gates
quality-gate: ## Run quality gate checks (must pass for CI/CD)
	@echo "$(BLUE)Running quality gate checks...$(NC)"
	$(MAKE) lint
	$(MAKE) test-coverage
	$(MAKE) security
	@echo "$(GREEN)✓ All quality gates passed!$(NC)"

pre-commit: ## Run all pre-commit checks
	@echo "$(BLUE)Running pre-commit checks...$(NC)"
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-fast
	@echo "$(GREEN)✓ Pre-commit checks passed!$(NC)"

# Release management
version-bump-patch: ## Bump patch version (1.0.0 -> 1.0.1)
	@echo "$(BLUE)Bumping patch version...$(NC)"
	bump2version patch

version-bump-minor: ## Bump minor version (1.0.0 -> 1.1.0)
	@echo "$(BLUE)Bumping minor version...$(NC)"
	bump2version minor

version-bump-major: ## Bump major version (1.0.0 -> 2.0.0)
	@echo "$(BLUE)Bumping major version...$(NC)"
	bump2version major

changelog: ## Generate changelog
	@echo "$(BLUE)Generating changelog...$(NC)"
	@if command -v git-changelog >/dev/null 2>&1; then \
		git-changelog -o CHANGELOG.md; \
	else \
		echo "$(YELLOW)git-changelog not installed. Install with: pip install git-changelog$(NC)"; \
	fi

# Maintenance commands
deps-audit: ## Audit dependencies for security vulnerabilities
	@echo "$(BLUE)Auditing dependencies...$(NC)"
	safety check --json || true
	pip-audit --format=json || true

deps-licenses: ## Check dependency licenses
	@echo "$(BLUE)Checking dependency licenses...$(NC)"
	@if command -v pip-licenses >/dev/null 2>&1; then \
		pip-licenses --format=table --order=license; \
	else \
		echo "$(YELLOW)pip-licenses not installed. Install with: pip install pip-licenses$(NC)"; \
	fi

stats: ## Show project statistics
	@echo "$(BLUE)Project Statistics:$(NC)"
	@echo "Lines of code:"
	@find $(PACKAGE_NAME) -name "*.py" -exec wc -l {} + | tail -1
	@echo "Test files:"
	@find $(TEST_DIR) -name "*.py" | wc -l
	@echo "Python files:"
	@find . -name "*.py" -not -path "./.tox/*" -not -path "./build/*" -not -path "./.venv/*" | wc -l

# CI/CD helpers
ci-install: ## Install dependencies for CI environment
	@echo "$(BLUE)Installing CI dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install tox

ci-test: ## Run tests suitable for CI environment
	@echo "$(BLUE)Running CI tests...$(NC)"
	$(TOX) -e py$(shell python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

ci-deploy: ## Deploy in CI environment
	@echo "$(BLUE)Deploying in CI...$(NC)"
	$(MAKE) build
	$(MAKE) publish-test

# Development server commands
serve-jupyter: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook server...$(NC)"
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

serve-lab: ## Start JupyterLab server
	@echo "$(BLUE)Starting JupyterLab server...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Utility commands
list-outdated: ## List outdated packages
	@echo "$(BLUE)Checking for outdated packages...$(NC)"
	$(PIP) list --outdated --format=table

update-requirements: ## Update requirements.txt with current environment
	@echo "$(BLUE)Updating requirements.txt...$(NC)"
	$(PIP) freeze > requirements-dev.txt
	@echo "$(GREEN)requirements-dev.txt updated!$(NC)"

show-deps: ## Show dependency tree
	@echo "$(BLUE)Dependency tree:$(NC)"
	@if command -v pipdeptree >/dev/null 2>&1; then \
		pipdeptree; \
	else \
		echo "$(YELLOW)pipdeptree not installed. Install with: pip install pipdeptree$(NC)"; \
	fi

# Help for specific workflows
help-dev: ## Show development workflow help
	@echo "$(BLUE)Development Workflow:$(NC)"
	@echo "1. $(YELLOW)make dev-setup$(NC)     - Setup development environment"
	@echo "2. $(YELLOW)make install-hooks$(NC) - Install git hooks"
	@echo "3. $(YELLOW)make test$(NC)          - Run tests during development"
	@echo "4. $(YELLOW)make lint$(NC)          - Check code quality"
	@echo "5. $(YELLOW)make format$(NC)        - Format code"
	@echo "6. $(YELLOW)make pre-commit$(NC)    - Run pre-commit checks"

help-release: ## Show release workflow help
	@echo "$(BLUE)Release Workflow:$(NC)"
	@echo "1. $(YELLOW)make quality-gate$(NC)     - Ensure quality"
	@echo "2. $(YELLOW)make version-bump-*$(NC)   - Bump version"
	@echo "3. $(YELLOW)make changelog$(NC)        - Update changelog"
	@echo "4. $(YELLOW)make release$(NC)          - Prepare release"
	@echo "5. $(YELLOW)make publish-test$(NC)     - Test on TestPyPI"
	@echo "6. $(YELLOW)make publish$(NC)          - Publish to PyPI"

help-ci: ## Show CI/CD workflow help
	@echo "$(BLUE)CI/CD Workflow:$(NC)"
	@echo "1. $(YELLOW)make ci-install$(NC)    - Install CI dependencies"
	@echo "2. $(YELLOW)make quality-gate$(NC)  - Run quality checks"
	@echo "3. $(YELLOW)make ci-test$(NC)       - Run CI tests"
	@echo "4. $(YELLOW)make ci-deploy$(NC)     - Deploy (if needed)"

# Check if required tools are installed
check-tools: ## Check if required development tools are installed
	@echo "$(BLUE)Checking development tools...$(NC)"
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)✗ Python not found$(NC)"; exit 1; }
	@echo "$(GREEN)✓ Python found$(NC)"
	@command -v $(PIP) >/dev/null 2>&1 || { echo "$(RED)✗ pip not found$(NC)"; exit 1; }
	@echo "$(GREEN)✓ pip found$(NC)"
	@command -v git >/dev/null 2>&1 || { echo "$(RED)✗ git not found$(NC)"; exit 1; }
	@echo "$(GREEN)✓ git found$(NC)"
	@command -v $(TOX) >/dev/null 2>&1 || { echo "$(YELLOW)? tox not found (run 'pip install tox')$(NC)"; }
	@command -v $(TOX) >/dev/null 2>&1 && echo "$(GREEN)✓ tox found$(NC)"
	@echo "$(GREEN)Development tools check complete!$(NC)"

.PHONY: open-docs open-coverage docker-build docker-test docker-shell
.PHONY: t ta l f c b d i
.PHONY: test-watch test-debug test-parallel test-random
.PHONY: docs-api docs-linkcheck quality-gate pre-commit
.PHONY: version-bump-patch version-bump-minor version-bump-major changelog
.PHONY: deps-audit deps-licenses stats ci-install ci-test ci-deploy
.PHONY: serve-jupyter serve-lab list-outdated update-requirements show-deps
.PHONY: help-dev help-release help-ci check-tools