import os
import cv2

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from config import _util
from collections import deque
from config import _detect, _constants

from deep_sort_realtime.deepsort_tracker import DeepSort

device = "cuda:2" if torch.cuda.is_available() else "cpu"
print("Thiết bị đang được sử dụng:", device)

# model = YOLO(model_path).to(device)

vehicles = _constants.VEHICLES
vehicles_car = _constants.CAR

buffer_size = _constants.BUFFER_SIZE

# 1. Khởi tạo mô hình YOLO và BoxAnnotator
coco_model = YOLO(_constants.COCO_MODEL).to(device)
detect_license_plate = YOLO(_constants.LICENSE_PLATE_MODEL).to(device)
detect_light = YOLO(_constants.TRAFFIC_LIGHT_MODEL).to(device)

# tracker = DeepSort(max_age=30, polygon=True)
tracker = sv.ByteTrack(lost_track_buffer=30)

# Load classname từ file classes.names
with open("./file_path/classes.names") as f:
    class_names = f.read().strip().split("\n")

colors = np.random.randint(0, 255, size=(len(class_names), 3))
def run_detection():
    """
    Hàm chính để thực hiện nhận diện trên webcam và hiển thị kết quả.
    """
    violating_vehicles = {}
    vehicles_info = {}
    captured_vehicles = {}

    # existing_traffic_lights = {}

    violating_vehicle_ids = set()  # Lưu danh sách track_id xe vi phạm

    polygon = _constants.POLYZONE
    box_annotator, label_annotator, polygon_zone, polygon_zone_annotator = (
        _detect.initialize_and_annotators(polygon)
    )

    frame_nmr = -1
    frame_expiration = _constants.FRAME_EXPIRATION

    cap = cv2.VideoCapture(_constants.VIDEO_DETECT)

    existing_traffic_light_buffer = deque(maxlen=buffer_size)

    output_folder = _constants.LICENSE_IMAGES
    os.makedirs(output_folder, exist_ok=True)

    # 3. Vòng lặp chính để đọc khung hình từ webcam
    while True:
        ret, frame = cap.read()

        frame_nmr += 1
        if not ret:
            print("Không nhận được khung hình (frame). Kết nối có thể đã bị ngắt.")
            return frame
        else:
            detection_lights = _detect.objects_tracking(frame, detect_light)
            traffic_lights, existing_traffic_light_buffer= _detect.detect_traffic_light(
                detection_lights, existing_traffic_light_buffer
            )
            if not traffic_lights:
                traffic_lights = {"default": "Unknown"}
            is_red_light = traffic_lights.get(1, "Unknown") == "Red"

            detection_results = []
            detection_results_update_tracker = []
            labels = []

            detections_vehicles = _detect.detect_objects(
                frame, coco_model, classes=vehicles
            )

            detections_vehicles = tracker.update_with_detections(detections=detections_vehicles)
            # print("det: ", detections_vehicles)

            # print(polygon_zone.current_count)

            for vehicle in detections_vehicles:
                xyxy, _, conf, class_id, track_id, class_name_dict = (
                    vehicle  # 0: green, 1:red, 2:yellow
                )
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                class_name = class_name_dict.get("class_name", None)

                label = f"#{track_id} {class_name} {conf:.2f}"
                # label = f"#{class_name} {conf:.2f}"
                labels.append(label)
                # detection_results.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
                detection_results.append([x1, y1, x2, y2, class_id, track_id])
                # captured_vehicles[track_id] = frame_nmr

            # tracks = tracker.update_tracks(detection_results, frame=frame)

            is_detections_in_zone = polygon_zone.trigger(detections_vehicles)

            # for track in tracks:
            #     if not track.is_confirmed():
            #         continue  # Bỏ qua track chưa đủ ổn định
            #     track_id = track.track_id
            #     class_id = track.get_det_class()
            #     bbox = track.to_tlbr()  # (x1, y1, x2, y2)
            #     # class_namess = class_names[class_id]
            #     x1_track, y1_track, x2_track, y2_track = bbox[0], bbox[1], bbox[2], bbox[3]
            #     detection_results_update_tracker.append(
            #         [x1_track, y1_track, x2_track, y2_track, class_id, track_id]
            #     )

            # # Detect license plates
            license_plates = _detect.detect_objects(frame, detect_license_plate)
            # print("Detection: ", license_plates)
            for license_plate_index, license_plate in enumerate(license_plates):
                xyxy_car, _, conf_license, class_id_license, _, _ = license_plate
                x1_license, y1_license, x2_license, y2_license = (
                    xyxy_car[0],
                    xyxy_car[1],
                    xyxy_car[2],
                    xyxy_car[3],
                )

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, class_id, car_id = _util.get_car(
                    license_plate, detection_results
                )

                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[
                        int(y1_license) : int(y2_license),
                        int(x1_license) : int(x2_license),
                        :,
                    ]

                    # # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(
                        license_plate_crop, cv2.COLOR_BGR2GRAY
                    )
                    license_plate_text, license_plate_text_score = None, 0
                    if class_id == 3:
                        # Read license plate number
                        license_plate_text, license_plate_text_score = (
                            _util.read_license_plate_motobike(license_plate_crop_gray)
                        )
                    elif class_id in vehicles_car:
                        # Read license plate number
                        license_plate_text, license_plate_text_score = (
                            _util.read_license_plate_car(license_plate_crop_gray)
                        )
                    # Nếu xe chưa có trong danh sách, khởi tạo dữ liệu
                    if car_id not in vehicles_info:
                        vehicles_info[car_id] = {
                            "class_id": class_id,
                            "bbox_car": [xcar1, ycar1, xcar2, ycar2],
                            "bbox_plate": None,  # Chưa có biển số
                            "plate_text": None,  # Chưa OCR được
                            "plate_score": 0.0,  # Giá trị mặc định
                            "track_id": car_id
                        }

                    if (
                        license_plate_text
                        and license_plate_text.strip()
                        and license_plate_text_score
                        > vehicles_info[car_id]["plate_score"]
                    ):
                        vehicles_info[car_id]["plate_text"] = license_plate_text
                        vehicles_info[car_id]["plate_score"] = license_plate_text_score
                        vehicles_info[car_id]["bbox_plate"] = [
                            x1_license,
                            y1_license,
                            x2_license,
                            y2_license,
                        ]

            annotated_frame = _detect.draw_boxes(
                frame,
                detections_vehicles,
                labels,
                box_annotator,
                label_annotator,
                polygon_zone_annotator,
            )

            for vehicle, in_zone in zip(detections_vehicles, is_detections_in_zone):
                xyxy, _, conf, class_id, track_id, _ = vehicle
                if (
                    in_zone
                    and track_id in vehicles_info
                    and vehicles_info[track_id]["plate_text"]
                ):
                    is_zone = is_red_light
                    # Nếu xe chưa có trong danh sách vi phạm, khởi tạo và thêm is_zone
                    if track_id not in violating_vehicles:
                        violating_vehicles[track_id] = vehicles_info[track_id].copy()
                        violating_vehicles[track_id]["is_zone"] = is_zone
                        violating_vehicle_ids.add(track_id)

            expired_ids = [
                tid
                for tid, last_frame in captured_vehicles.items()
                if frame_nmr - last_frame > frame_expiration
            ]
            for tid in expired_ids:
                print(f"Xóa track_id {tid} do đã quá {frame_expiration} frame.")
                del captured_vehicles[tid]
                if tid in vehicles_info:
                    del vehicles_info[tid]
                    violating_vehicle_ids.remove(tid)

            # 6. Hiển thị khung hình
            cv2.imshow("YOLOv8 - RTMP Stream", annotated_frame)

            # Nhấn phím ESC (mã ASCII 27) để thoát khỏi cửa sổ
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# Chạy chương trình
if __name__ == "__main__":
    run_detection()
