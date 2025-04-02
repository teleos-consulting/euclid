# Makefile for Euclid project

.PHONY: help install test test-unit test-integration test-docker lint clean

help:
	@echo "Available commands:"
	@echo "  make install         Install Euclid in development mode"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests"
	@echo "  make test-docker     Run tests in Docker environment"
	@echo "  make lint            Run code style checks"
	@echo "  make clean           Clean up temporary files"

install:
	pip install -e .

test: test-unit

test-unit:
	bash tests/run_unit_tests.sh

test-docker:
	docker-compose up --build

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --statistics

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf test_results/
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".coverage" -delete
	find . -name "htmlcov" -type d -exec rm -rf {} +
