from ultralytics import YOLO
from config import _constants


def train():
    model = YOLO("yolo11n.pt")

    data_path = _constants.DATA_PATH

    train_results = model.train(
        data=data_path, epochs=200, device=2, lr0=0.0015, optimizer="SGD", patience=50
    )
    
    return train_results
