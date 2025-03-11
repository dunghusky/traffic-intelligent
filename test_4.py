from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="engine", device="cuda:2", half=True)
