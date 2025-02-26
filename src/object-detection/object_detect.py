import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Uses the pre-trained YOLOv8 model

# Define objects to detect (smartphones, books, etc.)
TARGET_OBJECTS = ["cell phone", "book"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)[0]

    for obj in results.boxes.data:
        x1, y1, x2, y2, score, class_id = obj.tolist()
        label = results.names[int(class_id)]

        if label in TARGET_OBJECTS and score > 0.5:  # Confidence threshold
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()