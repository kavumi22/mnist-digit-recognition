import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class DigitRecognitionModel:

    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),

            layers.GlobalAveragePooling2D(),

            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def get_model_summary(self):
        return self.model.summary()

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def predict(self, image):
        if image.shape != (28, 28):
            raise ValueError("Image must be 28x28 pixels")

        image = image.astype('float32') / 255.0
        image = image.reshape(1, 28, 28, 1)

        predictions = self.model.predict(image, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit]

        return predicted_digit, predictions[0]