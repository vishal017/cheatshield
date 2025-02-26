from pynput import keyboard
import pygetwindow as gw
import time

# List of restricted keys
RESTRICTED_KEYS = {keyboard.Key.alt, keyboard.Key.cmd, keyboard.Key.tab, keyboard.Key.esc, keyboard.Key.f4}

# Function to detect window switching
def detect_window_minimization():
    active_window = gw.getActiveWindow()
    if active_window and "proctor" not in active_window.title.lower():
        print("⚠️ Warning: Exam window is minimized or switched!")

# Function to detect shortcut key presses
def on_press(key):
    if key in RESTRICTED_KEYS:
        print(f"⚠️ Unauthorized key press detected: {key}")

# Start listening for key presses
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Monitoring system for unauthorized actions...")

while True:
    detect_window_minimization()
    time.sleep(2)  # Check window every 2 seconds
