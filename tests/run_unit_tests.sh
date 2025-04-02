#!/bin/bash
set -e

# Define colors for output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color

echo -e "${YELLOW}Running Euclid unit tests...${NC}"

# Run unit tests
python -m unittest discover -s tests -p "test_*.py"

# If we get here, all tests passed
echo -e "${GREEN}All unit tests passed!${NC}"
