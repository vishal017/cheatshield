import cv2
import dlib
import numpy as np

# Load face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        nose = (landmarks.part(30).x, landmarks.part(30).y)  # Nose tip
        chin = (landmarks.part(8).x, landmarks.part(8).y)    # Chin
        left_cheek = (landmarks.part(2).x, landmarks.part(2).y)
        right_cheek = (landmarks.part(14).x, landmarks.part(14).y)

        # Estimate movement based on the relative positions of facial landmarks
        horizontal_movement = abs(left_cheek[0] - right_cheek[0])
        vertical_movement = abs(nose[1] - chin[1])

        if horizontal_movement > 100 or vertical_movement > 50:  # Adjust thresholds based on testing
            cv2.putText(frame, "Abnormal Head Movement", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Head Movement Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
