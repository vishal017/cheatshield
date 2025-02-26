import pyaudio
import numpy as np
import librosa
import tensorflow as tf

# Load pre-trained audio classification model (example: custom trained model)
MODEL_PATH = "models/audio_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Audio Recording Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 0.02  # Adjust for noise sensitivity

# Start Recording
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

print("Listening for unauthorized sounds...")

while True:
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16) / 32768.0  # Normalize
    if np.abs(data).mean() > THRESHOLD:
        # Extract Features (MFCCs)
        mfccs = librosa.feature.mfcc(y=data, sr=RATE, n_mfcc=13).T
        mfccs = np.expand_dims(mfccs, axis=0)  # Reshape for model input

        # Predict if unauthorized sound is detected
        prediction = model.predict(mfccs)
        if prediction[0][1] > 0.7:  # Assuming class 1 is "Unauthorized Speech"
            print("⚠️ Unauthorized sound detected!")

# Cleanup
stream.stop_stream()
stream.close()
audio.terminate()
