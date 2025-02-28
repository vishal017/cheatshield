import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
def load_dataset(data_dir):
    images = []
    labels = []
    for label, class_name in enumerate(["open", "close"]):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            image = cv2.resize(image, (24, 24))  # Resize to 24x24
            image = image / 255.0  # Normalize to [0, 1]
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# Define the CNN model
def create_eye_blink_model(input_shape=(24, 24, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification (open/close)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_eye_blink_model():
    # Load the dataset
    data_dir = "data/face_images/eye_blink"
    images, labels = load_dataset(data_dir)

    # Reshape images for CNN input
    images = np.expand_dims(images, axis=-1)  # Add channel dimension

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_eye_blink_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained model
    model.save("models/eye_blink_model.h5")
    print("Eye blink detection model saved to models/eye_blink_model.h5")

if __name__ == "__main__":
    train_eye_blink_model()