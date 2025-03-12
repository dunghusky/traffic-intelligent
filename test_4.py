from ultralytics import YOLO
from config import _constants

model = YOLO(_constants.LICENSE_PLATE_MODEL)
model.export(
    format="/home/ubuntu/mekongai/test_intelligent/traffic-intelligent/model/model_export/model_tensorRT/license.engine",
    device="cuda:2",
    half=True,
)
