#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from model import DigitRecognitionModel
import tensorflow as tf
import os

def load_sample_data():
    print("Loading MNIST test dataset...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    sample_images = x_test[:10]
    sample_labels = y_test[:10]

    return sample_images, sample_labels

def find_latest_model():
    if not os.path.exists("saved_models"):
        return None

    model_files = [f for f in os.listdir("saved_models")
                  if f.endswith(('.keras', '.h5'))]

    if not model_files:
        return None

    model_paths = [os.path.join("saved_models", f) for f in model_files]
    latest_model = max(model_paths, key=os.path.getmtime)

    return latest_model

def visualize_predictions(images, true_labels, predictions, confidence_scores):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('MNIST Digit Recognition Results', fontsize=16)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        predicted = predictions[i]
        true_label = true_labels[i]
        confidence = confidence_scores[i][predicted] * 100

        color = 'green' if predicted == true_label else 'red'

        title = f'True: {true_label}, Pred: {predicted}\nConf: {confidence:.1f}%'
        ax.set_title(title, color=color, fontsize=10)

    plt.tight_layout()
    plt.show()

def print_detailed_results(true_labels, predictions, confidence_scores):
    print("\n" + "="*60)
    print("DETAILED PREDICTION RESULTS")
    print("="*60)

    correct = 0
    for i in range(len(predictions)):
        true_label = true_labels[i]
        predicted = predictions[i]
        confidence = confidence_scores[i][predicted] * 100

        status = "✓ CORRECT" if predicted == true_label else "✗ INCORRECT"

        print(f"Sample {i+1:2d}: True={true_label}, Predicted={predicted}, "
              f"Confidence={confidence:5.1f}% - {status}")

        if predicted == true_label:
            correct += 1

    accuracy = correct / len(predictions) * 100
    print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{len(predictions)})")

def show_confidence_distribution(confidence_scores, predictions):
    confidences = [confidence_scores[i][predictions[i]] for i in range(len(predictions))]

    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.axvline(np.mean(confidences), color='red', linestyle='--',
                label=f'Mean: {np.mean(confidences):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    print("MNIST Digit Recognition - Example Usage")
    print("="*50)

    model_path = find_latest_model()
    if model_path is None:
        print("ERROR: No trained model found!")
        print("Please train a model first by running: python train.py")
        return

    print(f"Loading model: {os.path.basename(model_path)}")

    try:
        model = DigitRecognitionModel()
        model.load_model(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    sample_images, sample_labels = load_sample_data()
    print(f"✓ Loaded {len(sample_images)} sample images")

    print("\nMaking predictions...")
    predictions = []
    all_confidence_scores = []

    for i, image in enumerate(sample_images):
        predicted_digit, confidence_scores = model.predict(image)
        predictions.append(predicted_digit)
        all_confidence_scores.append(confidence_scores)
        print(f"  Sample {i+1}: Predicted {predicted_digit} "
              f"(confidence: {confidence_scores[predicted_digit]*100:.1f}%)")

    print_detailed_results(sample_labels, predictions, all_confidence_scores)

    print("\nDisplaying visualization...")
    visualize_predictions(sample_images, sample_labels, predictions, all_confidence_scores)

    show_confidence_distribution(all_confidence_scores, predictions)

    print("\nExample completed! Check the plots for visual results.")

if __name__ == "__main__":
    main()