# Euclid Test Suite

This directory contains tests for verifying Euclid functionality in a Docker environment.

## Overview

The test suite runs Euclid against a real Ollama instance in a containerized environment, ensuring that all features work as expected.

## Running Tests

To run the tests:

```bash
docker-compose up --build
```

This will:

1. Start an Ollama container
2. Build the Euclid container
3. Run the test suite against the Ollama instance
4. Output test results to the `test_results` directory

## Test Categories

- **Basic Functionality**: Tests for model listing, basic queries
- **Tool Tests**: Tests for built-in tools like ls, cat, etc.
- **Function Calling**: Tests for function invocation pattern
- **Batch Tool**: Tests for parallel function execution

## Adding New Tests

To add new tests, modify `run_tests.sh` and add your test to the `TESTS` array:

```bash
TESTS=(
  "existing_test:command"
  "your_new_test:euclid command --options"
)
```

## Test Results

Test results are stored in the `test_results` directory with one file per test. Each file contains:

- The command that was executed
- The command output
- Test status (PASSED/FAILED)
