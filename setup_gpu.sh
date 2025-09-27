#!/bin/bash

# GPU Setup Script for MNIST Digit Recognition
# This script sets up CUDA support for TensorFlow on Arch Linux

set -e

echo "========================================================"
echo "GPU Setup - MNIST Digit Recognition"
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

# Check GPU
echo "Step 1: Checking GPU..."
if nvidia-smi &>/dev/null; then
    print_green "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    print_red "✗ No NVIDIA GPU detected"
    exit 1
fi
echo

# Install system CUDA packages
echo "Step 2: Installing CUDA packages..."
print_yellow "You will be prompted for your password..."

# Check if packages are already installed
if pacman -Q cuda cudnn &>/dev/null; then
    print_green "✓ CUDA packages already installed"
else
    echo "Installing: cuda, cudnn, python-pycuda..."
    sudo pacman -S --needed cuda cudnn python-pycuda || {
        print_red "Failed to install CUDA packages"
        print_yellow "Try manually: sudo pacman -S cuda cudnn python-pycuda"
        exit 1
    }
fi
echo

# Set environment variables
echo "Step 3: Setting up environment..."

# Add CUDA to PATH and LD_LIBRARY_PATH
export CUDA_HOME=/opt/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Make environment permanent by adding to bashrc if not already there
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA Environment Variables" >> ~/.bashrc
    echo "export CUDA_HOME=/opt/cuda" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    print_green "✓ Added CUDA environment variables to ~/.bashrc"
else
    print_green "✓ CUDA environment variables already in ~/.bashrc"
fi
echo

# Test TensorFlow GPU
echo "Step 4: Testing TensorFlow GPU support..."

# Activate virtual environment
source venv/bin/activate

# Test script
python << 'EOF'
import tensorflow as tf
import sys

print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU devices:", len(tf.config.list_physical_devices('GPU')))

if tf.config.list_physical_devices('GPU'):
    gpu = tf.config.list_physical_devices('GPU')[0]
    print("GPU details:", gpu)
    
    # Test GPU computation
    with tf.device('/GPU:0'):
        a = tf.random.uniform([1000, 1000])
        b = tf.random.uniform([1000, 1000])
        c = tf.matmul(a, b)
    print("✓ GPU computation test passed!")
    
    # Memory growth settings
    try:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except:
        print("Memory growth configuration skipped")
        
else:
    print("⚠️  No GPU detected by TensorFlow")
    print("This might be due to:")
    print("- Missing CUDA libraries")
    print("- Version mismatch between CUDA and TensorFlow")
    print("- Environment variables not set")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_green "✓ TensorFlow GPU test passed!"
else
    print_red "✗ TensorFlow GPU test failed"
    print_yellow "Continuing anyway - CPU training will work"
fi
echo

echo "========================================================"
print_green "GPU SETUP COMPLETED!"
echo "========================================================"
echo
echo "Next steps:"
print_yellow "1. Restart your terminal or run: source ~/.bashrc"
print_yellow "2. Activate venv: source venv/bin/activate"  
print_yellow "3. Train model: python train.py --epochs 1500 --batch-size 64 --model-name gpu_model"
echo
echo "Training should now be MUCH faster on GPU!"
echo "Expected speed improvement: 5-20x faster"
