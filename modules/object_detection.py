import cv2
import torch
import numpy as np
from yolov5 import YOLOv5  # Import the official YOLOv5 package

class ObjectDetector:
    def __init__(self):
        # Path to your custom-trained weights
        self.weights_path = r"models\best.pt"
        self.conf_thres = 0.2  # Lowered default confidence threshold for better book detection
        self.mobile_conf_thres = 0.4  # Lowered confidence threshold for better mobile phone detection
        self.iou_thres = 0.45    # IoU threshold for NMS
        self.classes = ['book', 'mobile phone', 'laptop']
        
        # Initialize the YOLOv5 model using the official package
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLOv5(model_path=self.weights_path, device=self.device)
        self.model.conf = self.conf_thres  # Set default confidence threshold
        self.model.iou = self.iou_thres    # Set IoU threshold
        self.alerts = {"objects": ""}

    def process_image(self, frame):
        # Convert frame to RGB (YOLOv5 expects RGB images)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference using the YOLOv5 model
        results = self.model.predict(img)
        
        # Extract detections
        h, w = frame.shape[:2]
        suspicious_objects = []
        
        # Process the results
        if results.pred[0].shape[0] > 0:  # If there are detections
            for det in results.pred[0]:
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = self.classes[int(cls)]
                
                # Apply higher confidence threshold for mobile phones
                if class_name == 'mobile phone' and conf < self.mobile_conf_thres:
                    print(f"Ignored mobile phone detection (Confidence: {conf:.2f} < {self.mobile_conf_thres})")
                    continue
                
                # Draw bounding box and label on the frame
                color = (0, 255, 0) if class_name != 'mobile phone' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                print(f"Detected object: {class_name} (Confidence: {conf:.2f})")  # Debug print
                if class_name in ['mobile phone', 'book', 'laptop']:
                    suspicious_objects.append(f"{class_name} (Confidence: {conf:.2f})")

        # Set alerts if suspicious objects are detected
        if suspicious_objects:
            self.alerts["objects"] = f"Abnormal Movement Detected: {', '.join(suspicious_objects)}"
        else:
            self.alerts["objects"] = ""

        return frame