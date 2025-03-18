from copy import deepcopy
import csv
import os
from typing import List
import cv2
import numpy as np

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from ultralytics import YOLO
import supervision as sv
from ultralytics.utils.plotting import Annotator
from collections import deque
from config import _detect, _constants

from utils_dt.general import find_in_list, load_zones_config
from utils_dt.timers import ClockBasedTimer


vehicles = _constants.VEHICLES
vehicles_car = _constants.CAR

buffer_size = _constants.BUFFER_SIZE

# 1. Khởi tạo mô hình YOLO và BoxAnnotator

detect_license_plate = YOLO(_constants.LICENSE_PLATE_MODEL)
detect_light = YOLO(_constants.TRAFFIC_LIGHT_MODEL)

box_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1)

class CustomSink:

    def __init__(self):
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = _constants.POLYZONE
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = sv.PolygonZone(
            polygon=self.polygons,
            triggering_anchors=(sv.Position.CENTER,),
        )
        # Thêm annotator để vẽ vùng polygon
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.zones, color=(0, 255, 0)
        )  # Màu xanh lá

    def on_prediction(self, detections: sv.Detections, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = detections
        print("\ntest: ", detections)

        annotated_frame = frame.image.copy()

        # detections_in_zone = self.zones.trigger(detections)
        # time_in_zone = self.timers[0].tick(detections_in_zone)
        # custom_color_lookup = np.full(detections_in_zone.class_id.shape, 0)

        # detection_results = []
        # labels = []
        # xyxy, _, conf, class_id, track_id, class_name_dict = (
        #     vehicle  # 0: green, 1:red, 2:yellow
        # )
        # x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        # class_name = class_name_dict.get("class_name", None)

        # label = f"#{track_id} {class_name} {conf:.2f}"
        # labels.append(label)
        # detection_results.append([x1, y1, x2, y2, class_id, track_id])

        # annotated_frame = box_annotator.annotate(
        #     detections=detections, scene=annotated_frame
        # )

        # annotated_frame = label_annotator.annotate(
        #     labels=labels, scene=annotated_frame, detections=detections
        # )
        # annotated_frame = self.zone_annotator.annotate(scene=annotated_frame)

        cv2.imshow("Processed Video", annotated_frame)
        cv2.waitKey(1)


def main(
    rtsp_url: str,
) -> None:
    coco_model = YOLO(_constants.COCO_MODEL)

    def inference_callback(frame: VideoFrame) -> sv.Detections:
        if isinstance(frame, list):  # Kiểm tra xem frame có phải là danh sách không
            frame = frame[0]  # Lấy phần tử đầu tiên của danh sách
        results = coco_model.track(
            frame.image, persist=True, conf=0.1, classes=vehicles
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        # print("\ntest_1: ", detections)
        return detections

    sink = CustomSink()

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=rtsp_url,
        on_video_frame=inference_callback,
        on_prediction=sink.on_prediction,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()


if __name__ == "__main__":
    main(
        rtsp_url="rtmp://34.223.48.131:1256/live",
    )
