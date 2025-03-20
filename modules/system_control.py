import keyboard
import pyautogui
import time

class SystemController:
    def __init__(self):
        self.is_test_active = False
        self.hotkeys = ['ctrl+c', 'ctrl+v', 'cmd+c', 'cmd+v']  # Key combinations to block

    def suppress_hotkey(self):
        # This function will be called when a blocked hotkey is pressed, and it will suppress the action
        return False

    def start_test(self):
        self.is_test_active = True
        print("Test started. Copy-paste is disabled, and fullscreen mode is enforced.")

        # Block copy-paste shortcuts using add_hotkey
        for hotkey in self.hotkeys:
            keyboard.add_hotkey(hotkey, self.suppress_hotkey, suppress=True)

        # Enforce fullscreen mode (using pyautogui to simulate F11)
        pyautogui.press('f11')

    def stop_test(self):
        self.is_test_active = False
        print("Test stopped. Copy-paste is re-enabled.")

        # Remove hotkey blocks
        for hotkey in self.hotkeys:
            keyboard.remove_hotkey(hotkey)

        # Exit fullscreen mode (optional, simulate F11 again to toggle)
        pyautogui.press('f11')

    def monitor_system(self):
        # Monitor for attempts to exit fullscreen (e.g., Alt+Tab, Esc)
        while self.is_test_active:
            if keyboard.is_pressed('alt+tab') or keyboard.is_pressed('esc'):
                print("Warning: Attempt to exit fullscreen detected!")
                pyautogui.press('f11')  # Re-enforce fullscreen
            time.sleep(0.1)

# Example usage (for testing)
if __name__ == "__main__":
    controller = SystemController()
    controller.start_test()

    try:
        controller.monitor_system()
    except KeyboardInterrupt:
        controller.stop_test()