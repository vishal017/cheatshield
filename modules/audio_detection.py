import pyaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

class AudioDetector:
    def __init__(self, sample_rate=16000, chunk_size=1024, detection_interval=1.0):
        # Audio capture parameters
        self.sample_rate = sample_rate  # YAMNet expects 16kHz audio
        self.chunk_size = chunk_size
        self.detection_interval = detection_interval  # How often to process audio (seconds)

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size)

        # Load YAMNet model
        print("Loading YAMNet model...")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_names = self._load_class_names()

        # Suspicious sound categories (relevant to cheating)
        self.suspicious_categories = [
            'Speech', 'Whispering', 'Conversation', 'Shout', 'Laughter',
            'Telephone', 'Music', 'Singing', 'Radio'
        ]

    def _load_class_names(self):
        # Load YAMNet class names from the AudioSet ontology
        # These are the 521 categories YAMNet can classify
        class_map_path = hub.resolve('https://tfhub.dev/google/yamnet/1') + '/assets/yamnet_class_map.csv'
        class_names = []
        with open(class_map_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                class_names.append(line.strip().split(',')[2])
        return class_names

    def detect_audio(self):
        # Read audio chunk
        audio_data = []
        for _ in range(int(self.sample_rate / self.chunk_size * self.detection_interval)):
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data.append(np.frombuffer(data, dtype=np.int16))
        audio_data = np.concatenate(audio_data)

        # Convert to float32 and normalize to [-1, 1] (YAMNet expects this format)
        audio_data = audio_data.astype(np.float32) / 32768.0

        # Ensure the audio is at least 0.96 seconds (YAMNet expects 0.96s frames)
        if len(audio_data) < self.sample_rate * 0.96:
            audio_data = np.pad(audio_data, (0, int(self.sample_rate * 0.96) - len(audio_data)))

        # Run YAMNet inference
        scores, _, _ = self.model(audio_data)
        scores = scores.numpy()

        # Get the top predicted class
        top_class_idx = np.argmax(scores[0])
        top_class = self.class_names[top_class_idx]
        confidence = scores[0][top_class_idx]

        # Check if the sound is suspicious
        is_suspicious = top_class in self.suspicious_categories and confidence > 0.5
        message = f"Detected: {top_class} (Confidence: {confidence:.2f})"
        if is_suspicious:
            message = f"Warning: Suspicious sound detected - {top_class} (Confidence: {confidence:.2f})"

        return is_suspicious, confidence, message

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# Example usage (for testing)
if __name__ == "__main__":
    detector = AudioDetector()
    print("Listening for suspicious sounds... (Press Ctrl+C to stop)")

    try:
        while True:
            is_suspicious, confidence, message = detector.detect_audio()
            print(message)
            time.sleep(0.1)
    except KeyboardInterrupt:
        detector.close()
        print("Audio detection stopped.")