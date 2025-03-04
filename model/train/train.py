from ultralytics import YOLO
from config import _constants


def train():
    model = YOLO("yolo11n.pt")

    data_path = _constants.DATA_PATH_LIGHT
    # data_path = "./traffic-intelligent/model/datasets/Vehicle-Registration-Plates-1/data.yaml"
    project = _constants.CHECKPOINT_PATH_1

    train_results = model.train(
        data=data_path, epochs=100, device=3, lr0=0.00015, optimizer="SGD", patience=50, project=project,
    )

    return train_results
