from copy import deepcopy
import os
import cv2
import torch

from ultralytics import YOLO
import supervision as sv
from ultralytics.utils.plotting import Annotator
from config import _util

# from paddleocr import PaddleOCR
from collections import deque

# Initialize the OCR reader
# reader = easyocr.Reader(["en"], gpu=False)
# ocr = PaddleOCR(lang="en", rec_algorithm="CRNN")
BUFFER_SIZE = 30
vehicles = [2, 3, 5, 7, 9]


def detect_objects(frame, model, conf=0.1, iou=0.5):
    """
    Nhận diện vật thể trên khung hình hiện tại.
    Args:
    - frame: Khung hình hiện tại từ video.
    - model: Mô hình YOLO đã được tải.

    Returns:
    - detections: Kết quả nhận diện.
    """
    # results = model(frame)[0]
    results = model.predict(frame, conf=conf, iou=iou)[0]
    detections = sv.Detections.from_ultralytics(results)
    return detections


def draw_boxes(frame, detections, box_annotator, lables_annatator):
    """
    Vẽ khung hộp và hiển thị nhãn lên khung hình.
    Args:
    - frame: Khung hình hiện tại.
    - detections: Đối tượng `Detections` từ YOLO.
    - box_annotator: Đối tượng để vẽ khung hộp.
    - model: Mô hình YOLO để lấy tên nhãn.

    Returns:
    - frame: Khung hình đã được vẽ khung hộp và nhãn.
    """

    labels = [
        f"#{tracker_id} {class_name} {confidence:.2f}"  # {confidence:.2f}
        for class_name, tracker_id, confidence in zip(  # , confidence
            detections["class_name"],
            detections.tracker_id,
            detections.confidence,
        )
    ]

    frame = box_annotator.annotate(detections=detections, scene=frame)

    frame = lables_annatator.annotate(labels=labels, scene=frame, detections=detections)

    return frame


def initialize_yolo_and_annotators(
    model_path: str, LINE_START: sv.Point, LINE_END: sv.Point
):
    """
    Khởi tạo mô hình YOLO và các annotator.
    """
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print("Thiết bị đang được sử dụng:", device)

    model = YOLO(model_path).to(device)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=1)
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line = sv.PolygonZone()
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=0, text_scale=0)
    byte_tracker = sv.ByteTrack()
    return (
        model,
        box_annotator,
        label_annotator,
        line_counter,
        line_annotator,
        byte_tracker,
    )


def detect_traffic_light(
    traffic_lights, existing_buffer=None
):  # (array([     1371.7,      274.18,      1432.8,      407.35], dtype=float32), None, 0.9041419, 0, 1, {'class_name': 'Green'})
    # Nếu đã có buffer, sử dụng nó, nếu không thì khởi tạo mới
    if existing_buffer is None:
        existing_buffer = deque(maxlen=BUFFER_SIZE)

    labels = []
    updated_states = {}
    for traffic_light in traffic_lights:
        xyxy, _, conf_license, class_id_license, track_id, class_name = (
            traffic_light  # 0: green, 1:red, 2:yellow
        )
        class_name = class_name.get("class_name", None)
        # x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        label = label = f"#{track_id} {class_name} {conf_license:.2f}"
        labels.append(label)

        updated_states[track_id] = class_name
        existing_buffer.append((track_id, class_name))

    # Nếu buffer trống (không có detection nào), trả về trạng thái mặc định hoặc None
    if len(existing_buffer) == 0:
        # Ví dụ: trả về trạng thái mặc định "Unknown"
        return {"default": "Unknown"}, existing_buffer

    for track_id, state in existing_buffer:
        if track_id not in updated_states:
            updated_states[track_id] = state

    return updated_states, existing_buffer, labels


def objects_tracking(frame, model, conf=0.1, iou=0.5):
    """
    Nhận diện vật thể trên khung hình hiện tại.
    Args:
    - frame: Khung hình hiện tại từ video.
    - model: Mô hình YOLO đã được tải.

    Returns:
    - detections: Kết quả nhận diện.
    """
    # results = model(frame)[0]
    results = model.track(frame, persist=True, conf=conf)[0]
    detections = sv.Detections.from_ultralytics(results)
    return detections


def run_detection():
    """
    Hàm chính để thực hiện nhận diện trên webcam và hiển thị kết quả.
    """

    # 1. Khởi tạo mô hình YOLO và BoxAnnotator
    coco_model = YOLO("yolo11n.pt")  # use to detect car and motorbike
    detect_license_plate = YOLO("./model/train/checkpoints/train/weights/best.pt")
    detect_light = YOLO("./model/train/checkpoints_lights/train/weights/best.pt")

    names = coco_model.model.names
    vehicles = [2, 3, 5, 7]

    results = {}
    # existing_traffic_lights = {}
    box_annatator = sv.BoxAnnotator(thickness=1)
    byte_tracker = sv.ByteTrack()
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)
    LINE_START = sv.Point(200, 0)
    LINE_END = sv.Point(200, 500)

    frame_nmr = -1

    # 2. Mở webcam và thiết lập các thông số
    cap = cv2.VideoCapture("./file_path/16h.mp4")

    existing_traffic_light_buffer = deque(maxlen=BUFFER_SIZE)

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
            detection_lights = objects_tracking(frame, detect_light)
            traffic_lights, existing_traffic_light_buffer, lab = detect_traffic_light(
                detection_lights, existing_traffic_light_buffer
            )
            if not traffic_lights:
                print(
                    "Không phát hiện được đèn giao thông, sử dụng trạng thái mặc định: Unknown"
                )
                traffic_lights = {"default": "Unknown"}
            print("Test: ", traffic_lights)

            # results[frame_nmr] = {}
            # detections_vehicles = objects_tracking(frame, coco_model)
            # # print("Detection: ", detections_vehicles)

            # detection_results = []
            # labels = []
            # for xyxy, confidence, class_id, track_id, class_name in zip(
            #     detections_vehicles.xyxy,
            #     detections_vehicles.confidence,
            #     detections_vehicles.class_id,
            #     detections_vehicles.tracker_id,
            #     detections_vehicles["class_name"]
            # ):
            #     x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            #     detection_results.append(
            #         [x1, y1, x2, y2, class_id, track_id]
            #     )

            #     label = f"#{track_id} {class_name} {confidence:.2f}"
            #     labels.append(label)

            annotated_frame = box_annatator.annotate(
                detections=detection_lights, scene=frame
            )

            annotated_frame = label_annotator.annotate(
                labels=lab, scene=frame, detections=detection_lights
            )

            # # Detect license plates
            # license_plates = detect_objects(frame, detect_license_plate)
            # # print("Detection: ", license_plates)
            # for license_plate_index, license_plate in enumerate(license_plates):
            #     xyxy_car, _, conf_license, class_id_license, _, _ = license_plate
            #     x1_license, y1_license, x2_license, y2_license = xyxy_car[0], xyxy_car[1], xyxy_car[2], xyxy_car[3]

            #     # assign license plate to car
            #     xcar1, ycar1, xcar2, ycar2, class_id, car_id = _util.get_car(
            #         license_plate, detection_results
            #     )

            #     if car_id != -1:
            #         # Crop license plate
            #         license_plate_crop = frame[
            #             int(y1_license) : int(y2_license),
            #             int(x1_license) : int(x2_license),
            #             :
            #         ]

            #         # # Lưu ảnh biển số xe đã cắt vào folder
            #         # file_name = f"frame_{frame_nmr}_plate_{license_plate_index}.png"
            #         # save_path = os.path.join(output_folder, file_name)
            #         # cv2.imwrite(save_path, license_plate_crop)
            #         # print(f"Saved cropped license plate: {save_path}")

            #         # # Process license plate
            #         license_plate_crop_gray = cv2.cvtColor(
            #             license_plate_crop, cv2.COLOR_BGR2GRAY
            #         )

            #         if class_id == 3:
            #             # Read license plate number
            #             license_plate_text, license_plate_text_score = (
            #                 _util.read_license_plate_motobike(license_plate_crop_gray)
            #             )
            #         elif class_id in vehicles_car:
            #             # Read license plate number
            #             license_plate_text, license_plate_text_score = (
            #                 _util.read_license_plate_car(license_plate_crop_gray)
            #             )

            #         if license_plate_text is not None:
            #             results[frame_nmr][car_id] = {
            #                 "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
            #                 "license_plate": {
            #                     "bbox": [
            #                         x1_license,
            #                         y1_license,
            #                         x2_license,
            #                         y2_license,
            #                     ],
            #                     "text": license_plate_text,
            #                     "bbox_score": conf_license,
            #                     "text_score": license_plate_text_score,
            #                 },
            #                 "class_id": class_id,
            #             }

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
