#!/bin/bash

# Documentation build script for EVoC

set -e  # Exit on any error

echo "Building EVoC Documentation"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "source/conf.py" ]; then
    echo "Error: Run this script from the doc directory"
    exit 1
fi

# Check if virtual environment exists, create if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing documentation requirements..."
pip install -r requirements.txt

# Install EVoC in development mode
echo "Installing EVoC in development mode..."
pip install -e ../.

# Clean previous build
echo "Cleaning previous build..."
make clean

# Build HTML documentation  
echo "Building HTML documentation..."
make html

# Check for warnings
if [ $? -eq 0 ]; then
    echo "Documentation built successfully!"
    echo "Open build/html/index.html in your browser to view"
else
    echo "Build failed with errors"
    exit 1
fi

# Optional: Run link check
if [ "$1" = "--check-links" ]; then
    echo "Checking links..."
    make linkcheck
fi

# Optional: Run doctests
if [ "$1" = "--test" ]; then
    echo "Running doctests..."
    make doctest
fi

echo "Build complete!"
