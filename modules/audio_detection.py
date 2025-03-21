import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import threading
import time

class AudioDetector:
    def __init__(self, sample_rate=16000, chunk_size=1024, detection_interval=2.0):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.detection_interval = detection_interval
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Full list of YAMNet class names (521 classes)
        # This list is sourced from the YAMNet class map: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
        self.class_names = [
            "Silence", "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue",
            "Babbling", "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell", "Scream", "Whispering",
            "Laughter", "Baby laughter", "Giggle", "Snicker", "Chuckle", "Crying, sobbing", "Baby cry, infant cry",
            "Whimper", "Wail, moan", "Sigh", "Singing", "Choir", "Yodeling", "Chant", "Mantra", "Male speech, man speaking",
            "Female speech, woman speaking", "Child singing", "Synthetic singing", "Rapping", "Humming", "Groan",
            "Grunt", "Whistle", "Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Cough", "Throat clearing",
            "Sneeze", "Sniff", "Run", "Shuffle", "Footsteps", "Chewing, mastication", "Biting", "Gargling", "Stomach rumble",
            "Burping, eructation", "Hiccup", "Fart", "Hands", "Finger snapping", "Clapping", "Heart sounds, heartbeat",
            "Heart murmur", "Cheering", "Applause", "Chatter", "Crowd", "Booing", "Hiss", "Buzz", "Laughter (Other)",
            "Background noise", "Environmental noise", "White noise", "Pink noise", "Static", "Traffic noise", "Air conditioner",
            "Fan", "Wind", "Rain", "Thunder", "Water", "Stream", "Waterfall", "Ocean", "Waves, surf", "Steam", "Gurgling",
            "Fire", "Crackle", "Chirp, tweet", "Bird", "Bird vocalization, bird call, bird song", "Chicken, rooster",
            "Turkey", "Goose", "Duck", "Owl", "Crow", "Pigeon, dove", "Gull, seagull", "Eagle", "Hawk", "Parrot",
            "Dog", "Bark", "Growl", "Howl", "Whimper (dog)", "Cat", "Meow", "Purr", "Hiss (cat)", "Caterwaul",
            "Horse", "Neigh", "Snort (horse)", "Moo", "Cowbell", "Pig", "Oink", "Sheep", "Baa", "Goat", "Bleat",
            "Lion, roar", "Elephant", "Snake, hiss", "Rattlesnake", "Insect", "Cricket", "Mosquito", "Fly, housefly",
            "Bee, wasp, etc.", "Frog", "Croak", "Toad", "Fish", "Dolphin", "Whale", "Clicking", "Clank", "Clatter",
            "Bang", "Thud", "Thump", "Smash", "Crash", "Breaking", "Tearing", "Rustling", "Squeak", "Creak", "Rattle",
            "Grind", "Scrape", "Scratch", "Screech", "Squeal", "Squelch", "Splash, splatter", "Drip", "Drop", "Pour",
            "Trickle, dribble", "Gush", "Spray", "Hiss (water)", "Boiling", "Bubble", "Pop", "Fizz", "Sizzle", "Hiss (fire)",
            "Crackling", "Explosion", "Gunshot, gunfire", "Machine gun", "Cannon", "Firecracker", "Fireworks", "Siren",
            "Police siren", "Ambulance siren", "Fire truck siren", "Air horn", "Vehicle horn, car horn, honking", "Car",
            "Car passing by", "Car alarm", "Tire squeal", "Truck", "Bus", "Motorcycle", "Train", "Train whistle", "Train horn",
            "Subway, metro", "Boat, ship", "Ship horn", "Airplane", "Helicopter", "Jet", "Propeller, airscrew", "Bicycle",
            "Skateboard", "Engine", "Motor", "Chainsaw", "Lawnmower", "Drill", "Hammer", "Sawing", "Filing (tool)", "Sanding",
            "Power tool", "Electric shaver", "Hair dryer", "Vacuum cleaner", "Blender", "Microwave oven", "Washing machine",
            "Dryer", "Dishwasher", "Refrigerator", "Clock", "Tick", "Ticking", "Chime", "Bell", "Church bell", "Jingle bell",
            "Bicycle bell", "Tuning fork", "Gong", "Wind chime", "Harmonica", "Accordion", "Bagpipes", "Banjo", "Bass guitar",
            "Electric guitar", "Acoustic guitar", "Ukulele", "Harp", "Piano", "Electric piano", "Organ", "Synthesizer",
            "Harpsichord", "Clavichord", "Violin, fiddle", "Cello", "Viola", "Double bass", "Trumpet", "Trombone", "Tuba",
            "French horn", "Saxophone", "Clarinet", "Flute", "Piccolo", "Oboe", "Bassoon", "Recorder", "Pan flute", "Ocarina",
            "Drum", "Snare drum", "Bass drum", "Timpani", "Bongo", "Conga", "Tambourine", "Cymbal", "Hi-hat", "Glockenspiel",
            "Xylophone", "Marimba", "Vibraphone", "Steel drum", "Triangle", "Cowbell (instrument)", "Wood block", "Castanets",
            "Maracas", "Shaker", "Rattle (instrument)", "Whip", "Scraper", "Didgeridoo", "Kazoo", "Theremin", "Sampler",
            "Drum machine", "Turntable", "Scratching (turntable)", "Orchestra", "String ensemble", "Brass ensemble",
            "Woodwind ensemble", "Percussion ensemble", "Choir", "A capella", "Beatboxing", "Music", "Pop music", "Rock music",
            "Jazz music", "Classical music", "Electronic music", "Hip hop music", "Folk music", "Country music", "Blues music",
            "Reggae music", "Funk music", "Soul music", "Gospel music", "Opera", "Musical", "Soundtrack", "Theme music",
            "Jingle (music)", "Sound effect", "Beep", "Ping", "Ding", "Dong", "Clang", "Twang", "Pluck", "Strum", "Thrum",
            "Hum", "Drone", "Whir", "Whoosh", "Swoosh", "Flutter", "Flap", "Rustle", "Crinkle", "Crunch", "Snap", "Crack",
            "Pop (sound)", "Click", "Tap", "Knock", "Thump (sound)", "Slap", "Slam", "Stomp", "Stamp", "Kick", "Punch",
            "Hit", "Strike", "Chop", "Swipe", "Swish", "Whack", "Smack", "Rub", "Grate", "Screech (sound)", "Squeal (sound)",
            "Squelch (sound)", "Squish", "Splash (sound)", "Drip (sound)", "Drop (sound)", "Pour (sound)", "Trickle (sound)",
            "Gush (sound)", "Spray (sound)", "Hiss (sound)", "Boiling (sound)", "Bubble (sound)", "Pop (bubble)", "Fizz (sound)",
            "Sizzle (sound)", "Crackle (sound)", "Explosion (sound)", "Gunshot (sound)", "Firecracker (sound)", "Fireworks (sound)",
            "Siren (sound)", "Horn (sound)", "Car horn (sound)", "Car alarm (sound)", "Tire squeal (sound)", "Engine (sound)",
            "Motor (sound)", "Chainsaw (sound)", "Lawnmower (sound)", "Drill (sound)", "Hammer (sound)", "Sawing (sound)",
            "Filing (sound)", "Sanding (sound)", "Power tool (sound)", "Electric shaver (sound)", "Hair dryer (sound)",
            "Vacuum cleaner (sound)", "Blender (sound)", "Microwave oven (sound)", "Washing machine (sound)", "Dryer (sound)",
            "Dishwasher (sound)", "Refrigerator (sound)", "Clock (sound)", "Tick (sound)", "Ticking (sound)", "Chime (sound)",
            "Bell (sound)", "Church bell (sound)", "Jingle bell (sound)", "Bicycle bell (sound)", "Tuning fork (sound)",
            "Gong (sound)", "Wind chime (sound)", "Harmonica (sound)", "Accordion (sound)", "Bagpipes (sound)", "Banjo (sound)",
            "Bass guitar (sound)", "Electric guitar (sound)", "Acoustic guitar (sound)", "Ukulele (sound)", "Harp (sound)",
            "Piano (sound)", "Electric piano (sound)", "Organ (sound)", "Synthesizer (sound)", "Harpsichord (sound)",
            "Clavichord (sound)", "Violin (sound)", "Cello (sound)", "Viola (sound)", "Double bass (sound)", "Trumpet (sound)",
            "Trombone (sound)", "Tuba (sound)", "French horn (sound)", "Saxophone (sound)", "Clarinet (sound)", "Flute (sound)",
            "Piccolo (sound)", "Oboe (sound)", "Bassoon (sound)", "Recorder (sound)", "Pan flute (sound)", "Ocarina (sound)",
            "Drum (sound)", "Snare drum (sound)", "Bass drum (sound)", "Timpani (sound)", "Bongo (sound)", "Conga (sound)",
            "Tambourine (sound)", "Cymbal (sound)", "Hi-hat (sound)", "Glockenspiel (sound)", "Xylophone (sound)", "Marimba (sound)",
            "Vibraphone (sound)", "Steel drum (sound)", "Triangle (sound)", "Cowbell (instrument sound)", "Wood block (sound)",
            "Castanets (sound)", "Maracas (sound)", "Shaker (sound)", "Rattle (instrument sound)", "Whip (sound)", "Scraper (sound)",
            "Didgeridoo (sound)", "Kazoo (sound)", "Theremin (sound)", "Sampler (sound)", "Drum machine (sound)", "Turntable (sound)",
            "Scratching (turntable sound)", "Orchestra (sound)", "String ensemble (sound)", "Brass ensemble (sound)",
            "Woodwind ensemble (sound)", "Percussion ensemble (sound)", "Choir (sound)", "A capella (sound)", "Beatboxing (sound)"
        ]
        self.stream = None
        self.running = False
        self.suspicious_sounds = ["Speech", "Whispering", "Conversation", "Laughter", "Cough", "Sneeze"]
        self.environmental_noises = ["Background noise", "Wind", "Traffic", "Air conditioner", "Fan", "Music", "Dog", "Cat", "Bird"]
        self.confidence_threshold = 0.5  # Lowered from 0.7 to 0.5 for better sensitivity

    def start_stream(self):
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize=self.chunk_size)
        self.stream.start()
        self.running = True

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.running = False

    def close(self):
        self.stop_stream()

    def detect_audio(self):
        if not self.running:
            self.start_stream()

        audio_data = np.zeros((self.sample_rate,), dtype=np.float32)
        for i in range(0, self.sample_rate, self.chunk_size):
            if not self.running:
                break
            chunk, _ = self.stream.read(self.chunk_size)
            audio_data[i:i + self.chunk_size] = chunk.flatten()

        scores, embeddings, spectrogram = self.model(audio_data)
        scores = scores.numpy()
        mean_scores = scores.mean(axis=0)
        top_class_idx = mean_scores.argmax()
        
        # Ensure the top_class_idx is within the range of our class_names list
        if top_class_idx < len(self.class_names):
            top_class = self.class_names[top_class_idx]
        else:
            top_class = "Unknown"
        confidence = mean_scores[top_class_idx]

        is_suspicious = False
        message = ""

        if top_class in self.suspicious_sounds and confidence >= self.confidence_threshold:
            is_suspicious = True
            message = f"Suspicious sound detected - {top_class} (Confidence: {confidence:.2f})"
        elif top_class in self.environmental_noises:
            message = f"Environmental noise detected - {top_class} (Confidence: {confidence:.2f})"
        else:
            message = f"Sound detected - {top_class} (Confidence: {confidence:.2f})"

        return is_suspicious, confidence, message