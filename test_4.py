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

    def on_prediction(self, detections: sv.Detections, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = detections

        annotated_frame = frame.image.copy()

        for idx, zone in enumerate(self.zones):
            annotated_frame = self.zones.annotate(scene=frame)

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = self.timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = box_annotator.annotate(detections=detections, scene=frame)

            annotated_frame = label_annotator.annotate(
                labels=labels, scene=frame, detections=detections
            )

        cv2.imshow("Processed Video", annotated_frame)
        cv2.waitKey(1)


def main(
    rtsp_url: str,
) -> None:
    coco_model = YOLO(_constants.COCO_MODEL)

    def inference_callback(frame: VideoFrame) -> sv.Detections:
        if isinstance(frame, list):  # Kiểm tra xem frame có phải là danh sách không
            frame = frame[0]  # Lấy phần tử đầu tiên của danh sách
        results = results = coco_model.track(
            frame.image, persist=True, conf=0.1, classes=vehicles
        )[0]
        return sv.Detections.from_ultralytics(results).with_nms(threshold=0.5)

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
