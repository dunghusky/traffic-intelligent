from ultralytics import YOLO
from config import _constants

model = YOLO(_constants.LICENSE_PLATE_MODEL_VPS)
model.export(
    format="engine",
    device="cuda:2",
    half=True,
)
