import numpy as np
import supervision as sv
# --------------------DATA PATH------------------------#
DATA_PATH = "/home/ubuntu/mekongai/test_intelligent/traffic-intelligent/model/datasets/Vehicle-Registration-Plates-2/data.yaml"
CHECKPOINT_PATH = "/home/ubuntu/mekongai/test_intelligent/traffic-intelligent/model/train/checkpoints_test"

DATA_PATH_LIGHT = "/home/ubuntu/mekongai/test_intelligent/traffic-intelligent/model/datasets/v5-1/data.yaml"
CHECKPOINT_PATH_1 = "/home/ubuntu/mekongai/test_intelligent/traffic-intelligent/model/train/checkpoints_lights"

# --------------------MODEL PATH------------------------#
COCO_MODEL = "yolo11n.pt"

LICENSE_PLATE_MODEL = "./model/train/checkpoints/train/weights/best.pt"

TRAFFIC_LIGHT_MODEL = "./model/train/checkpoints_lights/train/weights/best.pt"

# --------------------STORAGE PATH------------------------#
LICENSE_IMAGES = "./file_path/cropped_license_plates"

VIDEO_DETECT = "./file_path/16h15.5.9.22.mp4"

# --------------------POINT------------------------#
POINT_START = sv.Point(0, 861)
POINT_END = sv.Point(1292, 861)

# --------------------POLY ZONE------------------------#
POLYZONE = np.array(
    [
        [0, 750],  # Góc dưới bên trái
        [0, 913],  # Góc trên bên trái
        [1352, 832],  # Góc dưới bên phải
        [1221, 660],  # Góc trên bên phải
    ]
)

# --------------------SIZE NUMBER------------------------#
BUFFER_SIZE = 30

FRAME_EXPIRATION = 1000
# --------------------ARRAY CAR------------------------#
VEHICLES = [2, 3, 5, 7]

CAR = [2, 5, 7]
