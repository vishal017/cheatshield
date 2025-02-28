import librosa
import numpy as np
from tensorflow.keras import models
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the trained autoencoder model
def load_audio_anomaly_model():
    """
    Load the pre-trained audio anomaly detection model.
    """
    try:
        model = models.load_model("models/audio_anomaly_model.h5")
        logging.info("Audio anomaly detection model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load audio anomaly detection model: {e}")
        return None

# Extract MFCC features from audio
def extract_features(audio_path, n_mfcc=13):
    """
    Extract MFCC features from an audio file.
    """
    try:
        y, sr = librosa.load(audio_path, duration=5)  # Load first 5 seconds of audio
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)  # Return the mean of MFCC features
    except Exception as e:
        logging.error(f"Failed to extract features from audio: {e}")
        return None

# Detect background noise
def detect_background_noise(audio_path, model, threshold=0.1):
    """
    Detect background noise using the trained autoencoder model.
    """
    try:
        features = extract_features(audio_path)
        if features is None:
            return False

        # Predict using the autoencoder model
        reconstruction_error = np.mean(np.square(features - model.predict([features])))
        logging.info(f"Reconstruction error: {reconstruction_error}")

        # If reconstruction error exceeds the threshold, flag as anomaly
        return reconstruction_error > threshold
    except Exception as e:
        logging.error(f"Failed to detect background noise: {e}")
        return False