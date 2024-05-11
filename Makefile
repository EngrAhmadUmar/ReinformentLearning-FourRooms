# Makefile for RL assignment

# Set up virtual environment
venv:
    python3 -m venv venv

# Install dependencies
install:
    . venv/bin/activate && pip install -r requirements.txt

# Run Scenario 1
scenario1:
    . venv/bin/activate && python Scenario1.py

# Run Scenario 2
scenario2:
    . venv/bin/activate && python Scenario2.py

# Run Scenario 3
scenario3:
    . venv/bin/activate && python Scenario3.py

# Run all scenarios
run-all: scenario1 scenario2 scenario3

# Clean up
clean:
    rm -rf venv

# Help
help:
    @echo "Available targets:"
    @echo "  venv           : Set up virtual environment"
    @echo "  install        : Install dependencies"
    @echo "  scenario1      : Run Scenario 1"
    @echo "  scenario2      : Run Scenario 2"
    @echo "  scenario3      : Run Scenario 3"
    @echo "  run-all        : Run all scenarios"
    @echo "  clean          : Clean up virtual environment"
    @echo "  help           : Show this help message"

# Default target
.DEFAULT_GOAL := help
