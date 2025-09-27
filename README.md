# MNIST Digit Recognition Neural Network

A complete implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset. This project includes training scripts, a graphical user interface for testing, and comprehensive documentation.

## 🌟 Features

- **Deep Learning Model**: Custom CNN architecture optimized for digit recognition
- **Training Pipeline**: Complete training script with 1000+ epochs support
- **Interactive GUI**: Draw digits on a 28×28 canvas and get real-time predictions
- **Model Management**: Save/load trained models with automatic checkpoint saving
- **Visualization**: Training history plots and confidence score displays
- **Easy Setup**: Simple installation and usage instructions

## 📊 Model Architecture

The neural network uses a Convolutional Neural Network (CNN) with the following architecture:

```
Input (28×28×1) 
    ↓
Conv2D (32 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
Flatten
    ↓
Dense (64 units) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (10 units) + Softmax
```

**Expected Performance**: >98% accuracy on MNIST test set after 1000 epochs

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/digit-recognition-app2.git
   cd digit-recognition-app2
   ```

2. **Install dependencies**:

   **For Windows/macOS**:
   ```bash
   pip install -r requirements.txt
   ```

   **For Linux (Recommended - using virtual environment)**:
   
   **Option 1: Automatic setup (recommended)**
   ```bash
   # Run the automated setup script
   ./setup_linux.sh
   ```
   
   **Note**: The script supports Ubuntu/Debian, CentOS/RHEL, Fedora, and Arch Linux.
   
   **Option 2: Manual setup**
   
   **For Ubuntu/Debian:**
   ```bash
   # Install dependencies
   sudo apt-get update
   sudo apt-get install python3-venv python3-tk
   
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate
   
   # Upgrade pip
   pip install --upgrade pip
   
   # Install dependencies
   pip install -r requirements.txt
   ```
   
   **For Arch Linux:**
   ```bash
   # Install dependencies (venv is included with python)
   sudo pacman -S python python-pip tk
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate
   
   # Upgrade pip
   pip install --upgrade pip
   
   # Install dependencies
   pip install -r requirements.txt
   ```
   
   **Note for Linux users**: Always activate the virtual environment before running any scripts:
   ```bash
   source venv/bin/activate
   ```
   
   To deactivate the virtual environment when you're done:
   ```bash
   deactivate
   ```

### Quick Start Commands

**For Linux users (with virtual environment):**

**Option 1: Automatic setup (recommended)**
```bash
# 1. Clone repository
git clone https://github.com/your-username/digit-recognition-app2.git
cd digit-recognition-app2

# 2. Run automated setup
./setup_linux.sh

# 3. Activate virtual environment (after setup completes)
source venv/bin/activate

# 4. Train the model (1000 epochs)
python train.py

# 5. Launch GUI for testing
python gui_app.py

# 6. See example usage
python example_usage.py

# When done, deactivate virtual environment
deactivate
```

**Option 2: Manual setup**
```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install python3-venv python3-tk

# 2. Clone repository
git clone https://github.com/your-username/digit-recognition-app2.git
cd digit-recognition-app2

# 3. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Test installation
python test_installation.py

# 6. Train the model (1000 epochs)
python train.py

# 7. Launch GUI for testing
python gui_app.py

# 8. See example usage
python example_usage.py

# When done, deactivate virtual environment
deactivate
```

**For Windows/macOS users:**
```bash
# 1. Clone repository
git clone https://github.com/your-username/digit-recognition-app2.git
cd digit-recognition-app2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test installation
python test_installation.py

# 4. Train the model (1000 epochs)
python train.py

# 5. Launch GUI for testing
python gui_app.py

# 6. See example usage
python example_usage.py
```

### Training the Model

To train the neural network with default settings (1000 epochs):

**Linux (remember to activate venv first):**
```bash
source venv/bin/activate  # Activate virtual environment
python train.py
```

**Windows/macOS:**
```bash
python train.py
```

**Training Options**:
```bash
# Custom training parameters
python train.py --epochs 1500 --batch-size 64 --validation-split 0.2

# Quick test training (fewer epochs)
python train.py --epochs 10 --model-name quick_test

# Help - see all available options
python train.py --help
```

**Training Parameters**:
- `--epochs`: Number of training epochs (default: 1000)
- `--batch-size`: Batch size for training (default: 128)
- `--validation-split`: Fraction of training data for validation (default: 0.1)
- `--model-name`: Custom name for saved model (default: timestamp-based)

### Testing the Model

After training, launch the GUI application to test your model:

**Linux (remember to activate venv first):**
```bash
source venv/bin/activate  # Activate virtual environment
python gui_app.py
```

**Windows/macOS:**
```bash
python gui_app.py
```

## 📱 Using the GUI Application

The GUI provides an intuitive interface for testing your trained model:

### Interface Overview

1. **Drawing Canvas**: 28×28 pixel drawing area (scaled up for visibility)
2. **Model Status**: Shows currently loaded model
3. **Prediction Results**: Displays predicted digit and confidence
4. **Confidence Scores**: Shows probability for each digit (0-9)
5. **Control Buttons**: Clear, predict, save, and load functions

### How to Use

1. **Load a Model**:
   - Click "Load Default" to use the most recent trained model
   - Or click "Load Model" to select a specific model file

2. **Draw a Digit**:
   - Click and drag on the black canvas to draw
   - Draw digits centered and reasonably large
   - Use white color (created by clicking and dragging)

3. **Get Predictions**:
   - Click "Predict Digit" to see the model's prediction
   - View confidence scores for all digits (0-9)
   - The predicted digit is highlighted in blue

4. **Additional Features**:
   - "Clear Canvas": Start over with a blank canvas
   - "Save Image": Save your drawing as a PNG file

### Tips for Best Results

- **Center your digits**: Draw in the middle of the canvas
- **Make digits clear**: Use bold strokes for better recognition
- **Size matters**: Draw digits large enough to fill a good portion of the canvas
- **Single digits only**: The model is trained to recognize one digit at a time

## 📁 Project Structure

```
digit-recognition-app2/
├── model.py              # Neural network model definition
├── train.py              # Training script with full pipeline
├── train_gpu.py          # GPU-optimized training script
├── gui_app.py            # GUI application for testing
├── example_usage.py      # Example script showing model usage
├── test_installation.py  # Installation verification script
├── setup_linux.sh        # Automated setup script for Linux
├── setup_gpu.sh          # GPU setup script for CUDA support
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore file
├── saved_models/         # Directory for trained models (created during training)
├── training_plots/       # Directory for training history plots (created during training)
├── logs/                # TensorBoard logs (created during GPU training)
└── venv/                # Virtual environment directory (Linux users)
```

## 🚀 GPU Acceleration (Recommended)

For significantly faster training, enable GPU support if you have an NVIDIA graphics card.

### GPU Setup (Linux with NVIDIA GPU)

**Prerequisites**: NVIDIA GPU with recent drivers

**Automatic Setup**:
```bash
# Run GPU setup script (will install CUDA packages)
./setup_gpu.sh
```

**Manual Setup** (Arch Linux):
```bash
# Install CUDA packages
sudo pacman -S cuda cudnn python-pycuda

# Set environment variables
echo 'export CUDA_HOME=/opt/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc  
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### GPU Training

**Use the GPU-optimized training script for best performance**:

```bash
# Activate virtual environment
source venv/bin/activate

# Train with GPU acceleration (much faster!)
python train_gpu.py --epochs 1000 --batch-size 128

# For maximum quality (RTX 3050 can handle this)
python train_gpu.py --epochs 2000 --batch-size 256 --model-name ultra_quality
```

**GPU Training Features**:
- ✅ **5-20x faster** training compared to CPU
- ✅ **Mixed precision** training for RTX cards
- ✅ **Optimized batch sizes** for GPU memory
- ✅ **TensorBoard logging** for monitoring
- ✅ **Automatic memory management**

**Performance Comparison**:
- **CPU Training**: 60-90 minutes for 1000 epochs
- **GPU Training**: 3-5 minutes for 1000 epochs ⚡

## 🔧 Advanced Usage

### Custom Model Training

You can modify the model architecture in `model.py`:

```python
# Edit the _build_model method in the DigitRecognitionModel class
def _build_model(self):
    model = models.Sequential([
        # Add your custom layers here
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # ... more layers
    ])
    return model
```

### Training Callbacks

The training script includes several callbacks for optimal training:

- **ModelCheckpoint**: Saves the best model based on validation accuracy
- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
- **EarlyStopping**: Stops training if validation loss doesn't improve

### Using Trained Models Programmatically

```python
from model import DigitRecognitionModel
import numpy as np

# Load a trained model
model = DigitRecognitionModel()
model.load_model('saved_models/your_model.keras')

# Predict on a 28x28 image
image = np.random.randint(0, 255, (28, 28))  # Replace with actual image
predicted_digit, confidence_scores = model.predict(image)

print(f"Predicted digit: {predicted_digit}")
print(f"Confidence: {confidence_scores[predicted_digit]:.2f}")
```

## 📈 Training Results

After training, you'll find:

1. **Saved Models**: In the `saved_models/` directory
   - `mnist_model_TIMESTAMP.keras`: Best model during training
   - `mnist_model_TIMESTAMP_final.keras`: Final model after all epochs

2. **Training Plots**: In the `training_plots/` directory
   - Training and validation accuracy curves
   - Training and validation loss curves

3. **Console Output**: Real-time training progress with:
   - Epoch-by-epoch metrics
   - Final test accuracy
   - Model save locations

## 🛠️ Troubleshooting

### Common Issues

1. **"No module named 'tensorflow'"**:
   
   **Linux users:**
   ```bash
   source venv/bin/activate  # Make sure virtual environment is activated
   pip install tensorflow>=2.12.0
   ```
   
   **Windows/macOS users:**
   ```bash
   pip install tensorflow>=2.12.0
   ```

2. **GUI not opening on Linux**:
   ```bash
   sudo apt-get install python3-tk
   ```

3. **Training very slow**:
   - Consider using a GPU-enabled version of TensorFlow
   - Reduce batch size if running out of memory
   - Use fewer epochs for testing

4. **Model not loading**:
   - Make sure the model file path is correct
   - Check that the model was saved successfully during training
   - Verify TensorFlow version compatibility

5. **Linux Virtual Environment Issues**:
   
   **Virtual environment not found:**
   ```bash
   # Make sure you created it first
   python3 -m venv venv
   ```
   
   **Command not found after activating venv:**
   ```bash
   # Make sure you activated the environment
   source venv/bin/activate
   # Check if packages are installed
   pip list
   ```
   
   **Permission denied errors:**
   ```bash
   # Don't use sudo with pip in virtual environment
   source venv/bin/activate
   pip install -r requirements.txt  # WITHOUT sudo
   ```

### Performance Tips

- **GPU Acceleration**: Install `tensorflow-gpu` if you have a compatible GPU
- **Memory Usage**: Reduce batch size if you encounter out-of-memory errors
- **Training Time**: Start with fewer epochs (e.g., 100) to test the pipeline

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

1. **Bug Reports**: Open an issue if you find any bugs
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests with improvements
4. **Documentation**: Help improve the documentation

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Thanks to Yann LeCun and the MNIST database creators
- **TensorFlow**: For providing the deep learning framework
- **Python Community**: For the excellent libraries and tools

## 📚 Additional Resources

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [Keras API Reference](https://keras.io/api/)

---

**Happy Learning! 🎉**

If you found this project helpful, please consider giving it a star ⭐ on GitHub!
