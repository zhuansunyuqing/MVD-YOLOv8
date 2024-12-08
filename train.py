from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n-ship.yaml")

# Display model information (optional)
model.model

results = model.train(data="Seaship.yaml", epochs=300, imgsz=640)