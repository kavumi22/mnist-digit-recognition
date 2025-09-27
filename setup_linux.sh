#!/bin/bash

# MNIST Digit Recognition - Linux Setup Script
# This script automates the installation process for Linux users

set -e  # Exit on any error

echo "========================================================"
echo "MNIST Digit Recognition - Linux Setup"
echo "========================================================"
echo

# Function to print colored output
print_green() {
    echo -e "\033[32m$1\033[0m"
}

print_yellow() {
    echo -e "\033[33m$1\033[0m"
}

print_red() {
    echo -e "\033[31m$1\033[0m"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_red "This script is designed for Linux systems only."
    print_yellow "For other operating systems, please follow manual installation instructions."
    exit 1
fi

# Step 1: Install system dependencies
echo "Step 1: Installing system dependencies..."
print_yellow "You may be prompted for your password (sudo)..."

if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip python3-tk
    print_green "✓ System dependencies installed successfully"
elif command -v yum &> /dev/null; then
    sudo yum install -y python3 python3-venv python3-pip tkinter
    print_green "✓ System dependencies installed successfully"
elif command -v dnf &> /dev/null; then
    sudo dnf install -y python3 python3-venv python3-pip python3-tkinter
    print_green "✓ System dependencies installed successfully"
elif command -v pacman &> /dev/null; then
    sudo pacman -S --noconfirm python python-pip tk
    print_green "✓ System dependencies installed successfully"
else
    print_red "Unsupported package manager. Please install manually:"
    print_yellow "  - python3 (with venv module)"
    print_yellow "  - python3-pip"
    print_yellow "  - python3-tk (or tkinter)"
    print_yellow ""
    print_yellow "Note: On some distributions like Arch Linux, python-venv"
    print_yellow "is included in the main python package."
    exit 1
fi
echo

# Step 2: Create virtual environment
echo "Step 2: Creating virtual environment..."

if [ -d "venv" ]; then
    print_yellow "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
print_green "✓ Virtual environment created"
echo

# Step 3: Activate virtual environment and install dependencies
echo "Step 3: Installing Python dependencies..."

# Source the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_green "✓ Python dependencies installed successfully"
else
    print_red "requirements.txt not found!"
    exit 1
fi
echo

# Step 4: Test installation
echo "Step 4: Testing installation..."
if python test_installation.py; then
    print_green "✓ Installation test passed!"
else
    print_red "✗ Installation test failed. Please check the errors above."
    exit 1
fi
echo

# Success message
echo "========================================================"
print_green "SETUP COMPLETED SUCCESSFULLY!"
echo "========================================================"
echo
echo "Next steps:"
echo "1. Activate the virtual environment:"
print_yellow "   source venv/bin/activate"
echo
echo "2. Train the model (this may take a while):"
print_yellow "   python train.py"
echo
echo "3. Test the model with GUI:"
print_yellow "   python gui_app.py"
echo
echo "4. When you're done, deactivate the virtual environment:"
print_yellow "   deactivate"
echo
echo "For more information, see README.md"
echo

# Deactivate virtual environment
deactivate
