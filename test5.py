from copy import deepcopy
import cv2
import torch

from ultralytics import YOLO
import supervision as sv
from ultralytics.utils.plotting import Annotator

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


def run_detection():
    """
    Hàm chính để thực hiện nhận diện trên webcam và hiển thị kết quả.
    """

    # 1. Khởi tạo mô hình YOLO và BoxAnnotator
    coco_model = YOLO("yolo11n.pt")  # use to detect car and motorbike
    detect_license_plate = YOLO("./model/train/checkpoints/train/weights/best.pt")

    names = coco_model.model.names
    vehicles = [2, 3, 5, 7, 9]

    box_annatator = sv.BoxAnnotator(thickness=2)
    lables_annatator = sv.LabelAnnotator(text_thickness=4, text_scale=1)
    byte_tracker = sv.ByteTrack()
    label_annotator = sv.LabelAnnotator(text_thickness=3, text_scale=1)
    box_annotator = sv.BoxAnnotator(thickness=2)

    # 2. Mở webcam và thiết lập các thông số
    cap = cv2.VideoCapture("./file_path/16h15.5.9.22.mp4")

    # 3. Vòng lặp chính để đọc khung hình từ webcam
    while True:
        ret, frame = cap.read()
        # if not ret:
        #     print("Không nhận được khung hình (frame). Kết nối có thể đã bị ngắt.")
        #     break
        if ret:
            results = coco_model.track(frame, persist=True, classes=vehicles)[0]
            detections_vehicles = sv.Detections.from_ultralytics(results)
            print("Detection: ", detections_vehicles)

            detection_results = []
            for xyxy, confidence, class_id, track_id in zip(
                detections_vehicles.xyxy,
                detections_vehicles.confidence,
                detections_vehicles.class_id,
                detections_vehicles.tracker_id,
            ):
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                detection_results.append(
                    [x1, y1, x2, y2, round(float(confidence), 3), track_id]
                )

            # # Detect license plates
            # license_plates = detect_objects(frame, detect_license_plate)
            # for license_plate in license_plates:
            #     x1, y1, x2, y2 = license_plate.xyxy

            annotated_frame = results.plot()

            # 6. Hiển thị khung hình
            cv2.imshow("YOLOv8 - RTMP Stream", annotated_frame)

            # Nhấn phím ESC (mã ASCII 27) để thoát khỏi cửa sổ
            if cv2.waitKey(30) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


# Chạy chương trình
if __name__ == "__main__":
    run_detection()
