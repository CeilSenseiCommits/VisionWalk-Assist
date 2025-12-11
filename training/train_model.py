# 1. Install libraries first
!pip install roboflow ultralytics

# 2. Download the dataset
from roboflow import Roboflow
rf = Roboflow(api_key="XXXXXXXXXXXXXXX") # Replace with your Roboflow API Key
project = rf.workspace("suryansh-stlog").project("door-detection-and-alerts")
version = project.version(1)
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")

from ultralytics import YOLO

# Load the model
model = YOLO("yolov8m.pt")

# Train using the dataset location we just found
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=200,
    imgsz=640,
    batch=8
)