import cv2
from mtcnn import MTCNN
import numpy as np

class FaceDetector:
    def __init__(self):
        # Initialize the MTCNN detector
        self.detector = MTCNN()

    def detect_faces(self, frame):
        # Convert frame to RGB (MTCNN expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.detector.detect_faces(frame_rgb)

        # Draw bounding boxes around detected faces
        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            if confidence > 0.9:  # Confidence threshold
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Face {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Return the number of faces and the annotated frame
        return len(faces), frame

# Example usage (for testing)
if __name__ == "__main__":
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        num_faces, frame = detector.detect_faces(frame)
        print(f"Number of faces detected: {num_faces}")

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()