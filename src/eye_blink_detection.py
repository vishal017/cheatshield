import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained eye blink detection model
def load_eye_blink_model():
    """
    Load the pre-trained eye blink detection model.
    """
    try:
        model = load_model("models/eye_blink_model.h5")
        print("Eye blink detection model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load eye blink detection model: {e}")
        return None

# Detect eye blink in a frame
def detect_eye_blink(frame, model):
    """
    Detect eye blink in the given frame using the trained model.
    """
    try:
        # Preprocess the frame (resize, grayscale, normalize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (24, 24))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)

        # Predict eye blink
        prediction = model.predict(input_data)
        eye_blink_detected = prediction[0][0] > 0.5  # Adjust threshold if needed

        return frame, eye_blink_detected
    except Exception as e:
        print(f"Failed to detect eye blink: {e}")
        return frame, False