import cv2
from mtcnn import MTCNN

# Initialize MTCNN for face detection
detector = MTCNN()

def detect_faces(frame, threshold=0.5):
    """
    Detect faces in the frame using MTCNN.
    """
    faces = detector.detect_faces(frame)
    faces_detected = 0
    for face in faces:
        if face['confidence'] > threshold:
            faces_detected += 1
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame, faces_detected