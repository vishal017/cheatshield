import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import threading

class AudioDetector:
    def __init__(self, sample_rate=16000, chunk_size=1024, detection_interval=5.0):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.detection_interval = detection_interval
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_names = ['Speech', 'Whispering']
        self.audio_buffer = []
        self.running = True
        self.lock = threading.Lock()

    def detect_audio(self):
        try:
            # Record audio
            audio = sd.rec(int(self.detection_interval * self.sample_rate), samplerate=self.sample_rate, channels=1, blocking=True)
            audio = audio.flatten()
            
            # Debug: Print audio data statistics
            print(f"Audio data - Mean: {np.mean(audio):.4f}, Max: {np.max(audio):.4f}, Min: {np.min(audio):.4f}")
            
            # Normalize audio to [-1, 1]
            audio = audio / np.max(np.abs(audio) + 1e-7)
            
            # Ensure audio is at 16kHz
            if len(audio) < self.sample_rate:
                audio = np.pad(audio, (0, self.sample_rate - len(audio)), 'constant')
            audio = audio[:self.sample_rate]
            
            # Run YAMNet model
            scores, _, _ = self.model(audio)
            scores = scores.numpy()
            
            # Average scores over time
            avg_scores = np.mean(scores, axis=0)
            
            # Check for suspicious sounds (speech or whispering)
            speech_idx = self.class_names.index('Speech')
            whisper_idx = self.class_names.index('Whispering')
            
            speech_confidence = avg_scores[speech_idx]
            whisper_confidence = avg_scores[whisper_idx]
            
            # Lowered threshold for speech detection
            if speech_confidence > 0.3 or whisper_confidence > 0.3:
                if speech_confidence > whisper_confidence:
                    return True, speech_confidence, f"Suspicious sound detected - Speech (Confidence: {speech_confidence:.2f})"
                else:
                    return True, whisper_confidence, f"Suspicious sound detected - Whispering (Confidence: {whisper_confidence:.2f})"
            
            return False, 0.0, ""
        
        except Exception as e:
            print(f"Error in audio detection: {e}")
            return False, 0.0, ""

    def close(self):
        self.running = False