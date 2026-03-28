.PHONY: install format lint test clean help

# Default Python interpreter
PYTHON := python3

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make format        Format code with black and isort"
	@echo "  make lint          Run linting (flake8, mypy)"
	@echo "  make test          Run tests with pytest"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make clean         Remove cache and build files"
	@echo "  make all           Format, lint, test"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

format:
	@echo "Formatting with black..."
	black . --line-length=100
	@echo "Organizing imports with isort..."
	isort . --profile black

lint:
	@echo "Running flake8..."
	flake8 . --max-line-length=100 --count --statistics || true
	@echo "Running mypy..."
	mypy . --ignore-missing-imports || true

test:
	pytest -v

test-cov:
	pytest -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.mypy_cache' -delete
	find . -type d -name 'htmlcov' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.coverage' -delete
	find . -type f -name '.coverage' -delete

all: format lint test
	@echo "All checks passed!"
