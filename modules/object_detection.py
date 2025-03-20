import torch
import cv2

class ObjectDetector:
    def __init__(self):
        self.weights_path = r"models\best.pt"
        self.img_size = 416  # Reduced from 640 to 416
        self.conf_thres = 0.286
        self.iou_thres = 0.45
        self.classes = ['book', 'mobile phone', 'laptop']
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights_path, force_reload=True)
        self.model.conf = self.conf_thres
        self.model.iou = self.iou_thres
        self.model.classes = None
        self.model.eval()
        self.alerts = {"objects": ""}

    def process_image(self, frame):
        frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        preds = results.xyxy[0].cpu().numpy()
        h, w = frame.shape[:2]
        scale_x, scale_y = w / self.img_size, h / self.img_size

        suspicious_objects = []
        for pred in preds:
            x1, y1, x2, y2, conf, cls = pred
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            label = f"{self.classes[int(cls)]} {conf:.2f}"
            color = (0, 255, 0) if self.classes[int(cls)] != 'mobile phone' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Flag mobile phones as suspicious
            if self.classes[int(cls)] == 'mobile phone':
                suspicious_objects.append(f"Mobile phone (Confidence: {conf:.2f})")

        # Update alerts
        if suspicious_objects:
            self.alerts["objects"] = f"Abnormal Movement Detected: {', '.join(suspicious_objects)}"
        else:
            self.alerts["objects"] = ""

        return frame