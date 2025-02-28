import cv2
import numpy as np

# Load YOLOv5 model
net = cv2.dnn.readNet("models/yolov5s.weights", "models/yolov5s.cfg")
with open("models/coco.names", "r") as f:
    classes = f.read().splitlines()

def detect_objects(frame, threshold=0.5):
    """
    Detect prohibited objects (e.g., smartphones, books) using YOLOv5.
    """
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    objects_detected = False
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold and classes[class_id] in ["cell phone", "book"]:
                objects_detected = True
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return frame, objects_detected