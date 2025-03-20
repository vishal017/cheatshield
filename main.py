import sys
import cv2
import numpy as np
import threading
import time
import webbrowser
import psutil
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from modules.object_detection import ObjectDetector
from modules.face_detection import FaceDetector
from modules.audio_detection import AudioDetector
from modules.system_control import SystemController

class EndTestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("End Test")
        self.setModal(True)
        layout = QVBoxLayout()
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(QLabel("Enter Password to End Test:"))
        layout.addWidget(self.password_input)
        
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        layout.addWidget(self.error_label)
        
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.check_password)
        layout.addWidget(submit_button)
        
        self.setLayout(layout)
        self.password = "default"  # Default password

    def check_password(self):
        entered_password = self.password_input.text()
        if entered_password == self.password:
            self.accept()  # Close dialog and return True
        else:
            self.error_label.setText("Incorrect password. Please try again.")

class MonitoringWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Online Cheating Prevention - Monitoring")
        self.setGeometry(0, 0, 300, 280)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Webcam feed
        self.webcam_label = QLabel()
        self.webcam_label.setFixedSize(280, 180)
        layout.addWidget(self.webcam_label)
        
        # Violation count
        self.violation_label = QLabel("Violations: 0/10")
        self.violation_label.setStyleSheet("font-size: 14px; color: red;")
        layout.addWidget(self.violation_label)
        
        # End Test button
        end_test_button = QPushButton("End Test")
        end_test_button.setStyleSheet("background-color: #ff4444; color: white; padding: 5px; font-size: 14px;")
        end_test_button.clicked.connect(self.end_test)
        layout.addWidget(end_test_button)
        
        # Detection modules
        self.object_detector = ObjectDetector()
        self.face_detector = FaceDetector()
        self.audio_detector = AudioDetector(sample_rate=16000, chunk_size=1024, detection_interval=2.0)
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        # Test state
        self.test_active = True
        self.warning_count = 0
        self.max_warnings = 10
        self.frame_counter = 0
        self.closing = False  # Flag to indicate test is closing
        
        # Open the test website in the default browser
        self.test_url = "https://ps.bitsathy.ac.in/"
        webbrowser.open(self.test_url)
        
        # Start system controls
        self.system_controller = SystemController()
        self.system_controller.start_test()
        
        # Start detection threads
        self.audio_thread = threading.Thread(target=self.audio_monitoring)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start object detection in a separate thread
        self.object_detection_thread = threading.Thread(target=self.object_detection_loop)
        self.object_detection_thread.daemon = True
        self.object_detection_thread.start()
        
        # Start webcam update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # 10 FPS (100ms interval)
        
        # Keep window on top
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.show()

    def update_frame(self):
        if not self.test_active or self.closing:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.frame_counter += 1
        
        # Process frame with face detection (every other frame)
        if self.frame_counter % 2 == 0:
            num_faces, frame = self.face_detector.detect_faces(frame)
            if num_faces > 1:
                self.display_warning("Abnormal Movement Detected: Multiple faces detected!")
        
        # Use the latest frame processed by object detection
        if hasattr(self, 'latest_object_frame'):
            frame = self.latest_object_frame
        
        # Resize frame for display
        frame = cv2.resize(frame, (280, 180))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.webcam_label.setPixmap(QPixmap.fromImage(image))

    def object_detection_loop(self):
        while self.test_active and not self.closing:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Process frame with object detection (every 3rd frame)
                    if self.frame_counter % 3 == 0:
                        frame = self.object_detector.process_image(frame)
                        if self.object_detector.alerts["objects"]:
                            self.display_warning(self.object_detector.alerts["objects"])
                    self.latest_object_frame = frame
            time.sleep(0.1)  # Match the 10 FPS rate

    def audio_monitoring(self):
        while self.test_active and not self.closing:
            is_suspicious, confidence, message = self.audio_detector.detect_audio()
            if is_suspicious:
                sound_type = message.split(' - ')[1]
                self.display_warning(f"Abnormal Movement Detected: Suspicious sound - {sound_type}")
            time.sleep(0.1)

    def display_warning(self, message):
        if not self.test_active or self.closing:
            return
        self.warning_count += 1
        self.violation_label.setText(f"Violations: {self.warning_count}/{self.max_warnings}")
        print(f"Warning {self.warning_count}/{self.max_warnings}: {message}")
        if self.warning_count >= self.max_warnings:
            self.end_test(automatic=True)

    def end_test(self, automatic=False):
        if self.closing:
            return
        
        self.closing = True  # Set flag to stop all processing
        self.test_active = False
        
        if automatic:
            self.cleanup()
            self.close_application()
            return
        
        dialog = EndTestDialog(self)
        if dialog.exec_():
            self.cleanup()
            self.close_application()
        else:
            self.closing = False
            self.test_active = True
            # Restart the timer and threads if the password is incorrect
            self.timer.start(100)
            self.audio_thread = threading.Thread(target=self.audio_monitoring)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            self.object_detection_thread = threading.Thread(target=self.object_detection_loop)
            self.object_detection_thread.daemon = True
            self.object_detection_thread.start()

    def cleanup(self):
        # Stop the timer immediately to prevent new frames
        self.timer.stop()
        
        # Stop system controls
        self.system_controller.stop_test()
        
        # Close audio detector
        self.audio_detector.close()
        
        # Release the camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clean up object detector (YOLOv5 model)
        if hasattr(self, 'object_detector'):
            del self.object_detector
            torch.cuda.empty_cache()  # Clear GPU memory if using GPU
        
        # Clean up face detector
        if hasattr(self, 'face_detector'):
            del self.face_detector
        
        # Close the browser
        self.close_browser()

    def close_browser(self):
        browser_names = ["chrome", "firefox", "edge", "safari", "opera"]
        for proc in psutil.process_iter(['name']):
            try:
                proc_name = proc.info['name'].lower()
                if any(browser in proc_name for browser in browser_names):
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    def close_application(self):
        QApplication.quit()

    def closeEvent(self, event):
        if self.test_active and not self.closing:
            event.ignore()
            self.end_test()
        else:
            self.cleanup()
            event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MonitoringWindow()
    sys.exit(app.exec_())