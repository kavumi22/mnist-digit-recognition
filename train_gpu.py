#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import DigitRecognitionModel
import os
import argparse
from datetime import datetime
import time

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

            print(f"✓ GPU setup completed: {len(gpus)} GPU(s) available")
            print(f"✓ Mixed precision enabled for faster training")

            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("⚠️  No GPU detected, using CPU")
        return False

def load_and_preprocess_data():
    print("Loading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    return x_train, y_train, x_test, y_test

def create_gpu_optimized_callbacks(model_save_path):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=8,
            min_lr=0.00001,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/gpu_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]

    return callbacks

def train_gpu_model(epochs=1000, batch_size=128, validation_split=0.1, model_name=None):
    print("=" * 60)
    print("GPU-OPTIMIZED MNIST DIGIT RECOGNITION TRAINING")
    print("=" * 60)

    gpu_available = setup_gpu()

    if gpu_available:
        batch_size = min(batch_size * 2, 256)
        print(f"GPU detected: Using batch size {batch_size}")
    else:
        print(f"CPU mode: Using batch size {batch_size}")

    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    print("\nCreating GPU-optimized model...")
    model = DigitRecognitionModel()

    model.get_model_summary()

    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"gpu_model_{timestamp}"

    model_save_path = f"saved_models/{model_name}.keras"
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    callbacks = create_gpu_optimized_callbacks(model_save_path)

    print(f"\nStarting GPU-optimized training:")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split}")
    print(f"Mixed precision: {'Enabled' if gpu_available else 'Disabled'}")
    print(f"Model save path: {model_save_path}")
    print()

    start_time = time.time()

    history = model.model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.model.evaluate(x_test, y_test, verbose=0)

    plot_save_path = f"training_plots/gpu_training_{model_name}.png"
    os.makedirs("training_plots", exist_ok=True)
    plot_gpu_training_history(history, plot_save_path, training_time, test_accuracy)

    final_model_path = f"saved_models/{model_name}_final.keras"
    model.save_model(final_model_path)

    print("\n" + "=" * 60)
    print("GPU TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Total epochs trained: {len(history.history['loss'])}")
    print(f"Best model: {model_save_path}")
    print(f"Final model: {final_model_path}")
    print(f"Training plots: {plot_save_path}")

    if gpu_available:
        print(f"GPU acceleration was used!")
    else:
        print("⚠️  Training used CPU (GPU not available)")

def plot_gpu_training_history(history, save_path, training_time, test_accuracy):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if 'lr' in history.history:
        ax3.plot(history.history['lr'], label='Learning Rate', color='orange', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Learning Rate\nNot Recorded', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Learning Rate Schedule')

    ax4.axis('off')
    summary_text = f"""
GPU Training Summary

Training Time: {training_time:.1f}s ({training_time/60:.1f} min)
Final Test Accuracy: {test_accuracy:.4f}
Total Epochs: {len(history.history['loss'])}
Best Val Accuracy: {max(history.history['val_accuracy']):.4f}

GPU Acceleration: {'✓ Used' if tf.config.list_physical_devices('GPU') else '✗ Not Used'}
Mixed Precision: {'✓ Enabled' if tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else '✗ Disabled'}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized MNIST Training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (will be auto-adjusted for GPU)')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--model-name', type=str, default=None, help='Model name')

    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    train_gpu_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()