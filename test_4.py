import torch
from ultralytics import YOLO
from config import _constants
import supervision as sv

device = "cuda:2" if torch.cuda.is_available() else "cpu"
print("Thiết bị đang được sử dụng:", device)
model = YOLO(
    "/home/ubuntu/mekongai/test_intelligent/traffic-intelligent/model/model_export/model_tensorRT/yolo11n.engine",
)

result = model.predict("https://ultralytics.com/images/bus.jpg")
print("\nresult: ", result)

detections = sv.Detections.from_ultralytics(result)
print("\ndetections: ", detections)
