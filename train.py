import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import DigitRecognitionModel
import os
import argparse
from datetime import datetime

def load_and_preprocess_data():
    print("Loading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test

def create_callbacks(model_save_path):
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
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]

    return callbacks

def plot_training_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")

def train_model(epochs=1000, batch_size=128, validation_split=0.1, model_name=None):
    print("=" * 60)
    print("MNIST Digit Recognition Model Training")
    print("=" * 60)

    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    print("\nCreating model...")
    digit_model = DigitRecognitionModel()
    digit_model.get_model_summary()

    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"mnist_model_{timestamp}"

    model_save_path = f"saved_models/{model_name}.keras"
    os.makedirs("saved_models", exist_ok=True)

    callbacks = create_callbacks(model_save_path)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split}")
    print(f"Model will be saved to: {model_save_path}")

    history = digit_model.model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = digit_model.model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    plot_save_path = f"training_plots/training_history_{model_name}.png"
    os.makedirs("training_plots", exist_ok=True)
    plot_training_history(history, plot_save_path)

    final_model_path = f"saved_models/{model_name}_final.keras"
    digit_model.save_model(final_model_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best model saved to: {model_save_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history plot: {plot_save_path}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Total epochs trained: {len(history.history['loss'])}")

def main():
    parser = argparse.ArgumentParser(description='Train MNIST Digit Recognition Model')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Validation split (default: 0.1)')
    parser.add_argument('--model-name', type=str, default=None, help='Custom model name (default: timestamp-based)')

    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()