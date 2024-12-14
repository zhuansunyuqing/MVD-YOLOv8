# MVD-YOLOv8
Marine Vessel Detection-YOLOv8 (MVD-YOLOv8) for improving the performance of marine vessel detection.

# Training
1. Download the datasets
>SeaShips: https: //github.com/jiaming-wang/SeaShips  
>Ship-Detection: https://universe.roboflow.com/yolo-fruit-team4/ship-detection-jxcyp  
>Ships/Vessels in Aerial Images: https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images?resource=download  
  
2. Load and train a YOLOv8n model
>The training code is as followed: 
```
model = YOLO("yolov8n-ship.yaml")
# Display model information (optional)
model.model
results = model.train(data="Seaship.yaml", epochs=300, imgsz=640)
```

# Some issues to know
1. The environment is Python 3.8. For more details, please refer to requirements.txt.
2. Default anchors are used. If you use your own anchors, probably some changes are needed.
