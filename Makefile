# Makefile for LLM Hook Analysis Framework

.PHONY: help install dev-install test lint format clean docs benchmark

help:
	@echo "Available commands:"
	@echo "  install      - Install package in production mode"
	@echo "  dev-install  - Install package in development mode with dev dependencies"
	@echo "  test         - Run test suite"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  docs         - Build documentation"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=llm_hooks --cov-report=html --cov-report=term

lint:
	flake8 llm_hooks/ --max-line-length=100 --extend-ignore=E203,W503
	mypy llm_hooks/ --ignore-missing-imports

format:
	black llm_hooks/ examples/ tutorials/ tests/ --line-length=100
	isort llm_hooks/ examples/ tutorials/ tests/ --profile black

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

benchmark:
	python scripts/benchmark.py --model small --iterations 100
	python scripts/benchmark.py --model medium --iterations 100
	python scripts/benchmark.py --model transformer --iterations 100

docs:
	@echo "Documentation is in Markdown format in docs/ directory"
	@echo "No build step required"

.DEFAULT_GOAL := help
