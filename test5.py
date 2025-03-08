from copy import deepcopy
import os
import cv2
import numpy as np

from ultralytics import YOLO
import supervision as sv
from ultralytics.utils.plotting import Annotator
from config import _util
from collections import deque
from config import _detect, _points_calculation


vehicles = [2, 3, 5, 7]

vehicles_car = [2, 5, 7]

BUFFER_SIZE = 30


def run_detection():
    """
    Hàm chính để thực hiện nhận diện trên webcam và hiển thị kết quả.
    """

    # 1. Khởi tạo mô hình YOLO và BoxAnnotator
    coco_model = YOLO("yolo11n.pt")  # use to detect car and motorbike
    detect_license_plate = YOLO("./model/train/checkpoints/train/weights/best.pt")
    detect_light = YOLO("./model/train/checkpoints_lights/train/weights/best.pt")

    results = {}
    # existing_traffic_lights = {}
    start = sv.Point(0, 861)
    end = sv.Point(1292, 861)
    box_annatator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)
    line_zone = sv.LineZone(start=start, end=end)
    line_ana = sv.LineZoneAnnotator(thickness=1)

    frame_nmr = -1
    FRAME_EXPIRATION = 100
    # Danh sách các xe đã cắt ảnh (track_id -> frame cuối cùng xe xuất hiện)
    captured_vehicles = {}

    # 2. Mở webcam và thiết lập các thông số
    cap = cv2.VideoCapture("./file_path/16h15.25.9.22.mp4")

    existing_traffic_light_buffer = deque(maxlen=BUFFER_SIZE)

    in_count = 0

    violating_vehicles = {}

    polygon = np.array(
        [
            [0, 750],  # Góc dưới bên trái
            [0, 913],  # Góc trên bên trái
            [1352, 832],  # Góc dưới bên phải
            [1221, 660],  # Góc trên bên phải
        ]
    )
    polygon_zone = sv.PolygonZone(polygon=polygon)
    polygon_zone_ana = sv.PolygonZoneAnnotator(zone=polygon_zone, thickness=1)

    output_folder = "./file_path/cropped_license_plates"
    os.makedirs(output_folder, exist_ok=True)

    # 3. Vòng lặp chính để đọc khung hình từ webcam
    while True:
        ret, frame = cap.read()

        frame_nmr += 1
        if not ret:
            print("Không nhận được khung hình (frame). Kết nối có thể đã bị ngắt.")
            return frame
        else:
            # detection_lights = objects_tracking(frame, detect_light)
            # traffic_lights, existing_traffic_light_buffer, lab = detect_traffic_light(
            #     detection_lights, existing_traffic_light_buffer
            # )
            # if not traffic_lights:
            #     print("Không phát hiện được đèn giao thông, sử dụng trạng thái mặc định: Unknown")
            #     traffic_lights = {"default": "Unknown"}
            # print("Test: ", traffic_lights)

            results[frame_nmr] = {}
            detections_vehicles = _detect.objects_tracking(
                frame, coco_model, classes=vehicles
            )
            # print("Detection: ", detections_vehicles)
            is_detections_in_zone = polygon_zone.trigger(
                detections_vehicles
            )
            print(polygon_zone.current_count)

            detection_results = []
            labels = []

            for vehicle, is_in_zone in zip(detections_vehicles, is_detections_in_zone):
                xyxy, _, conf, class_id, track_id, class_name_dict = (
                    vehicle  # 0: green, 1:red, 2:yellow
                )
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                class_name = class_name_dict.get("class_name", None)

                label = f"#{track_id} {class_name} {conf:.2f}"
                labels.append(label)

                if is_in_zone:
                    if track_id not in captured_vehicles:
                        car_crop = frame[
                            int(y1) : int(y2),
                            int(x1) : int(x2),
                            :
                        ]

                        _points_calculation.save_violation_image(
                            car_crop, track_id, frame_nmr, output_folder
                        )

                        # Cập nhật danh sách đã cắt ảnh (track_id, frame cuối thấy xe, kích thước bbox)
                        captured_vehicles[track_id] = frame_nmr

                        detection_results.append([x1, y1, x2, y2, class_id, track_id])

                    captured_vehicles[track_id] = captured_vehicles[track_id] = (
                        frame_nmr
                    )

            # # Detect license plates
            license_plates = _detect.detect_objects(frame, detect_license_plate)
            # print("Detection: ", license_plates)
            for license_plate_index, license_plate in enumerate(license_plates):
                xyxy_car, _, conf_license, class_id_license, _, _ = license_plate
                x1_license, y1_license, x2_license, y2_license = xyxy_car[0], xyxy_car[1], xyxy_car[2], xyxy_car[3]

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, class_id, car_id = _util.get_car(
                    license_plate, detection_results
                )

                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[
                        int(y1_license) : int(y2_license),
                        int(x1_license) : int(x2_license),
                        :
                    ]

                    # # Lưu ảnh biển số xe đã cắt vào folder
                    _points_calculation.save_violation_image(
                        license_plate_crop,
                        license_plate_index,
                        frame_nmr,
                        output_folder,
                    )

                    # # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(
                        license_plate_crop, cv2.COLOR_BGR2GRAY
                    )

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

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                            "license_plate": {
                                "bbox": [
                                    x1_license,
                                    y1_license,
                                    x2_license,
                                    y2_license,
                                ],
                                "text": license_plate_text,
                                "bbox_score": conf_license,
                                "text_score": license_plate_text_score,
                            },
                            "class_id": class_id,
                        }

            annotated_frame = box_annatator.annotate(
                detections=detections_vehicles, scene=frame
            )

            annotated_frame = label_annotator.annotate(
                labels=labels, scene=frame, detections=detections_vehicles
            )

            # annotated_frame = line_ana.annotate(frame=frame, line_counter=line_zone)
            annotated_frame = polygon_zone_ana.annotate(
                scene=frame
            )

            # Xóa track_id cũ không còn thấy sau X frame để tránh trùng ID
            expired_ids = [
                tid
                for tid, last_frame in captured_vehicles.items()
                if frame_nmr - last_frame > FRAME_EXPIRATION
            ]
            for tid in expired_ids:
                print(f"Xóa track_id {tid} do đã quá {FRAME_EXPIRATION} frame.")
                del captured_vehicles[tid]

            # 6. Hiển thị khung hình
            cv2.imshow("YOLOv8 - RTMP Stream", annotated_frame)
            _util.write_csv(results, "./z_test.csv")

            # Nhấn phím ESC (mã ASCII 27) để thoát khỏi cửa sổ
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# Chạy chương trình
if __name__ == "__main__":
    run_detection()
