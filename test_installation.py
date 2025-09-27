#!/usr/bin/env python3

import sys

def test_imports():
    print("Testing imports...")

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False

    try:
        from PIL import Image
        print(f"✓ Pillow (PIL) {Image.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False

    try:
        import tkinter as tk
        print("✓ tkinter imported successfully")
    except ImportError as e:
        print(f"✗ tkinter import failed: {e}")
        print("  On Ubuntu/Debian: sudo apt-get install python3-tk")
        return False

    return True

def test_model_creation():
    print("\nTesting model creation...")

    try:
        from model import DigitRecognitionModel
        model = DigitRecognitionModel()
        print("✓ Model created successfully")

        model.get_model_summary()
        print("✓ Model summary generated successfully")

        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_data_loading():
    print("\nTesting MNIST data loading...")

    try:
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        print(f"✓ MNIST data loaded successfully")
        print(f"  Training samples: {x_train.shape[0]}")
        print(f"  Test samples: {x_test.shape[0]}")
        print(f"  Image shape: {x_train.shape[1:3]}")

        return True
    except Exception as e:
        print(f"✗ MNIST data loading failed: {e}")
        return False

def test_gui_components():
    print("\nTesting GUI components...")

    try:
        import tkinter as tk
        from PIL import Image, ImageDraw

        root = tk.Tk()
        root.withdraw()

        image = Image.new("L", (28, 28), 0)
        draw = ImageDraw.Draw(image)

        root.destroy()

        print("✓ GUI components test passed")
        return True
    except Exception as e:
        print(f"✗ GUI components test failed: {e}")
        return False

def check_virtual_environment():
    import sys
    import os

    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if os.name == 'posix':
        if in_venv:
            print("✓ Running in virtual environment (recommended for Linux)")
        else:
            print("⚠️  Not running in virtual environment")
            print("   For Linux users, it's recommended to use: source venv/bin/activate")
    else:
        print("ℹ️  Virtual environment check (Linux-specific) - skipped on this OS")

def main():
    print("MNIST Digit Recognition - Installation Test")
    print("=" * 50)

    check_virtual_environment()
    print()

    tests = [
        ("Import Test", test_imports),
        ("Model Creation Test", test_model_creation),
        ("Data Loading Test", test_data_loading),
        ("GUI Components Test", test_gui_components)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n[{passed_tests + 1}/{total_tests}] {test_name}")
        print("-" * 40)

        if test_func():
            passed_tests += 1
        else:
            print(f"⚠️  {test_name} failed!")

    print("\n" + "=" * 50)
    print("INSTALLATION TEST SUMMARY")
    print("=" * 50)

    if passed_tests == total_tests:
        print(f"🎉 ALL TESTS PASSED! ({passed_tests}/{total_tests})")
        print("\nYour installation is ready!")
        print("Next steps:")
        print("1. Train the model: python train.py")
        print("2. Test with GUI: python gui_app.py")
        print("3. See examples: python example_usage.py")
    else:
        print(f"❌ {total_tests - passed_tests} test(s) failed! ({passed_tests}/{total_tests} passed)")
        print("\nPlease fix the failed tests before proceeding.")
        print("Try: pip install -r requirements.txt")

        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)