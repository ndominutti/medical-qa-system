#!/bin/bash

echo "Setting up your dev environment..."
# Install pre-commit hooks
pip3 install pre-commit
pre-commit install

echo "✅ Done. pre-commit hooks are installed."