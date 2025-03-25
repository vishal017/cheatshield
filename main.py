import sys
import cv2
import numpy as np
import threading
import time
import psutil
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QDialog
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView
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
        # Password should be set by the user in a configuration file or environment variable
        self.password = "SET_YOUR_PASSWORD"  # Placeholder; replace with your own password

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
        
        # Apply CSS styles to improve design (without changing position)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 10px;
            }
            QLabel#webcam_label {
                border: 2px solid #333;
                border-radius: 5px;
                background-color: #000;
            }
            QLabel#violation_label {
                font-size: 14px;
                color: #d32f2f;
                font-weight: bold;
                margin-top: 5px;
                margin-bottom: 5px;
            }
            QPushButton {
                background-color: #ff4444;
                color: white;
                padding: 5px;
                font-size: 14px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
        """)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Webcam feed
        self.webcam_label = QLabel()
        self.webcam_label.setObjectName("webcam_label")
        self.webcam_label.setFixedSize(280, 180)
        self.webcam_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.webcam_label)
        
        # Violation count
        self.violation_label = QLabel("Violations: 0/10")
        self.violation_label.setObjectName("violation_label")
        layout.addWidget(self.violation_label)
        
        # End Test button
        end_test_button = QPushButton("End Test")
        end_test_button.clicked.connect(self.end_test)
        layout.addWidget(end_test_button)
        
        # Web view for the test website (reverting to QWebEngineView)
        self.web_view = QWebEngineView()
        self.web_view.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        screen = QApplication.primaryScreen().size()
        self.web_view.setGeometry(0, 0, screen.width(), screen.height())  # Fullscreen
        # Replace with the actual test URL or configure via environment variable
        self.test_url = "https://example.com/"  # Placeholder; set your test URL
        self.web_view.load(QUrl(self.test_url))
        
        # Inject JavaScript to enforce fullscreen and disable shortcuts/right-click
        js_code = """
        // Enter fullscreen mode
        document.documentElement.requestFullscreen();
        
        // Disable right-click
        document.addEventListener('contextmenu', function(e) {
            e.preventDefault();
        });
        
        // Disable keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Block Windows+D (keyCode 68 is 'D')
            if (e.key === 'd' && e.metaKey) {
                e.preventDefault();
            }
            // Block F11 (exit fullscreen)
            if (e.key === 'F11') {
                e.preventDefault();
            }
            // Block Alt+Tab, Alt+F4, etc.
            if (e.altKey) {
                e.preventDefault();
            }
            // Block Ctrl+Shift+Esc (Task Manager)
            if (e.ctrlKey && e.shiftKey && e.key === 'Escape') {
                e.preventDefault();
            }
        });
        
        // Ensure fullscreen on focus
        window.addEventListener('blur', function() {
            setTimeout(() => {
                document.documentElement.requestFullscreen();
            }, 100);
        });
        """
        self.web_view.page().runJavaScript(js_code)
        self.web_view.show()
        
        # Alert overlay for warnings (reverting to QWebEngineView)
        self.alert_view = QWebEngineView()
        self.alert_view.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.alert_view.setAttribute(Qt.WA_TranslucentBackground)
        self.alert_view.setStyleSheet("background: transparent;")  # Ensure transparency
        self.alert_view.setFixedSize(800, 100)  # Wider to accommodate longer messages
        self.alert_view.setGeometry(screen.width() // 2 - 400, 50, 800, 100)  # Position near the top
        
        # Embed the HTML content with warning display logic
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Violation Alert</title>
            <style>
                body {
                    background: transparent !important;
                    margin: 0;
                    padding: 0;
                }
                #violation-alert {
                    position: fixed;
                    top: 10%;
                    left: 50%;
                    transform: translateX(-50%);
                    background-color: rgba(255, 0, 0, 0.9);
                    color: white;
                    padding: 15px 30px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                    z-index: 9999;
                    font-size: 18px;
                    font-family: Arial, sans-serif;
                    text-align: center;
                    max-width: 80%;
                    word-wrap: break-word;
                }
            </style>
        </head>
        <body>
            <script>
                function showViolationAlert(message) {
                    // Remove any existing alert
                    const existingAlert = document.getElementById('violation-alert');
                    if (existingAlert) {
                        existingAlert.remove();
                    }

                    // Create a div for the alert
                    const alertDiv = document.createElement('div');
                    alertDiv.id = 'violation-alert';
                    alertDiv.innerHTML = message;
                    document.body.appendChild(alertDiv);

                    // Remove the alert after 10 seconds
                    setTimeout(() => {
                        alertDiv.remove();
                    }, 10000);
                }

                function displayAlert(message) {
                    showViolationAlert(message);
                }
            </script>
        </body>
        </html>
        """
        # Removed personal file path; use a generic base URL
        self.alert_view.setHtml(html_content, QUrl("file://"))
        self.alert_view.hide()
        
        # Detection modules
        self.object_detector = ObjectDetector()
        self.face_detector = FaceDetector()
        self.audio_detector = AudioDetector(sample_rate=16000, chunk_size=1024, detection_interval=5.0)
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)  # Reduced resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        
        # Test state
        self.test_active = True
        self.warning_count = 0
        self.max_warnings = 10
        self.frame_counter = 0
        self.closing = False
        self.last_violation_time = 0
        self.violation_cooldown = 15  # 15 seconds cooldown between violations
        
        # Start system controls
        self.system_controller = SystemController()
        self.system_controller.start_test()
        
        # Start audio detection thread
        self.audio_thread = threading.Thread(target=self.audio_monitoring)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start webcam update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # 10 FPS (100ms interval)
        
        # Keep window on top and remove window controls
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.show()

    def update_frame(self):
        if not self.test_active or self.closing:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.frame_counter += 1
        
        # Process frame with object detection (every 8th frame)
        if self.frame_counter % 8 == 0:
            frame = self.object_detector.process_image(frame)
            if self.object_detector.alerts["objects"]:
                # Customize message for mobile phone detection
                if "mobile phone" in self.object_detector.alerts["objects"].lower():
                    self.display_warning("Warning: Mobile phone detected")
                else:
                    self.display_warning(self.object_detector.alerts["objects"])
        
        # Process frame with face detection (every 8th frame)
        if self.frame_counter % 8 == 0:
            num_faces, frame = self.face_detector.detect_faces(frame)
            if num_faces == 0:
                self.display_warning("Face not visible, please show your face")
            elif num_faces > 1:
                self.display_warning("Abnormal Movement Detected: Multiple faces detected!")
        
        # Resize frame for display
        frame = cv2.resize(frame, (280, 180))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.webcam_label.setPixmap(QPixmap.fromImage(image))
        
        # Ensure the monitoring window and alert view stay on top
        self.raise_()
        self.alert_view.raise_()

    def audio_monitoring(self):
        while self.test_active and not self.closing:
            is_suspicious, confidence, message = self.audio_detector.detect_audio()
            if is_suspicious:
                self.display_warning(message)
            time.sleep(self.audio_detector.detection_interval)

    def display_warning(self, message):
        if not self.test_active or self.closing:
            return
        
        # Check if enough time has passed since the last violation
        current_time = time.time()
        if current_time - self.last_violation_time < self.violation_cooldown:
            return  # Ignore the violation if within cooldown period
        
        # Increment violation count only for specific warnings
        if "Face not visible" not in message:  # Don't count "Face not visible" as a violation
            self.warning_count += 1
            self.last_violation_time = current_time
            self.violation_label.setText(f"Violations: {self.warning_count}/{self.max_warnings}")
            print(f"Warning {self.warning_count}/{self.max_warnings}: {message}")
        else:
            print(f"Warning (not counted): {message}")
        
        # Show JavaScript alert
        self.alert_view.show()
        self.alert_view.page().runJavaScript(f'displayAlert("{message}");')
        
        # Ensure the alert view stays on top
        self.alert_view.raise_()
        
        # Only end the test if the maximum warnings are reached
        if self.warning_count >= self.max_warnings:
            print("Maximum warnings reached. Ending test...")
            self.end_test(automatic=True)

    def focusInEvent(self, event):
        # Re-raise the window when it gains focus
        self.raise_()
        self.alert_view.raise_()
        super().focusInEvent(event)

    def end_test(self, automatic=False):
        if self.closing:
            return
        
        self.closing = True
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
            self.timer.start(100)
            self.audio_thread = threading.Thread(target=self.audio_monitoring)
            self.audio_thread.daemon = True
            self.audio_thread.start()

    def cleanup(self):
        self.timer.stop()
        self.system_controller.stop_test()
        self.audio_detector.close()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'object_detector'):
            del self.object_detector
            torch.cuda.empty_cache()
        if hasattr(self, 'face_detector'):
            del self.face_detector
        self.alert_view.hide()
        self.web_view.hide()
        self.close_browser()

    def close_browser(self):
        browser_names = ["chrome", "firefox", "edge", "safari", "opera", "brave"]
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