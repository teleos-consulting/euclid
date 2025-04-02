#!/bin/bash
set -e

# Define colors for output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color

echo -e "${YELLOW}Starting Euclid test suite...${NC}"

# Ensure Ollama service is ready
echo -e "${YELLOW}Checking Ollama service...${NC}"
until curl -s http://ollama:11434/api/tags > /dev/null; do
  echo "Waiting for Ollama service..."
  sleep 3
done

# Pull a small test model
echo -e "${YELLOW}Pulling test model...${NC}"
ollama pull tiny-llama:latest

# Create test directory
MKDIR -p /app/test_results

# Run tests
run_test() {
  local test_name=$1
  local command=$2
  local output_file="/app/test_results/${test_name}.log"
  
  echo -e "${YELLOW}Running test: ${test_name}${NC}"
  echo "Command: $command" > "$output_file"
  
  if eval "$command" >> "$output_file" 2>&1; then
    echo -e "${GREEN}✓ Test passed: ${test_name}${NC}"
    echo "TEST PASSED" >> "$output_file"
    return 0
  else
    echo -e "${RED}✗ Test failed: ${test_name}${NC}"
    echo "TEST FAILED" >> "$output_file"
    return 1
  fi
}

# Standard tests for Euclid functionality
TESTS=(
  "model_list:euclid models"
  "basic_query:euclid run 'Hello world'"
  "directory_list:euclid run '/ls /app'"
  "file_view:euclid run '/cat /app/README.md' --no-stream"
  "glob_search:euclid run '/GlobTool pattern=\"*.py\" path=\"/app\"' --no-stream"
  "grep_search:euclid run '/GrepTool pattern=\"def\" path=\"/app\" include=\"*.py\"' --no-stream"
)

# Run all tests
FAILED_TESTS=0
for test in "${TESTS[@]}"; do
  IFS=':' read -r name cmd <<< "$test"
  if ! run_test "$name" "$cmd"; then
    FAILED_TESTS=$((FAILED_TESTS + 1))
  fi
done

# Function tests
FUNCTION_TEST_FILE="/app/test_results/function_test.log"
echo -e "${YELLOW}Testing function calling${NC}"
echo "Testing function calling" > "$FUNCTION_TEST_FILE"

cat > /tmp/function_test_prompt.txt << 'EOL'
List all Python files in the /app directory using the GlobTool function.

Then, show me the first 10 lines of the README.md file using the View function.

Please execute these tasks using the function calling format.
EOL

if euclid run "$(cat /tmp/function_test_prompt.txt)" --model tiny-llama >> "$FUNCTION_TEST_FILE" 2>&1; then
  echo -e "${GREEN}✓ Test passed: function_calling${NC}"
  echo "TEST PASSED" >> "$FUNCTION_TEST_FILE"
else
  echo -e "${RED}✗ Test failed: function_calling${NC}"
  echo "TEST FAILED" >> "$FUNCTION_TEST_FILE"
  FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Check for BatchTool functionality
BATCH_TEST_FILE="/app/test_results/batch_tool_test.log"
echo -e "${YELLOW}Testing BatchTool${NC}"
echo "Testing BatchTool" > "$BATCH_TEST_FILE"

cat > /tmp/batch_test_prompt.txt << 'EOL'
Use BatchTool to run the following tasks in parallel:
1. List all Python files in the /app directory using GlobTool
2. Search for the term "function" in Python files using GrepTool

Please execute these tasks in a single function call using BatchTool.
EOL

if euclid run "$(cat /tmp/batch_test_prompt.txt)" --model tiny-llama >> "$BATCH_TEST_FILE" 2>&1; then
  echo -e "${GREEN}✓ Test passed: batch_tool${NC}"
  echo "TEST PASSED" >> "$BATCH_TEST_FILE"
else
  echo -e "${RED}✗ Test failed: batch_tool${NC}"
  echo "TEST FAILED" >> "$BATCH_TEST_FILE"
  FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Summary
if [ $FAILED_TESTS -eq 0 ]; then
  echo -e "${GREEN}All tests passed!${NC}"
  exit 0
else
  echo -e "${RED}${FAILED_TESTS} test(s) failed. Check logs in /app/test_results for details.${NC}"
  exit 1
fi
