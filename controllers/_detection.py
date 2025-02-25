from copy import deepcopy
import cv2
import torch


from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
import supervision as sv

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
            detections["class_name"], detections.tracker_id, detections.confidence,
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

    box_annatator = sv.BoxAnnotator(thickness=2)
    lables_annatator = sv.LabelAnnotator(text_thickness=4, text_scale=1)
    byte_tracker = BYTETracker()
    # bot_sort = sv.B

    # 2. Mở webcam và thiết lập các thông số
    cap = cv2.VideoCapture("./file_path/20221003-102700.mp4")

    # 3. Vòng lặp chính để đọc khung hình từ webcam
    while True:
        ret, frame = cap.read()
        # if not ret:
        #     print("Không nhận được khung hình (frame). Kết nối có thể đã bị ngắt.")
        #     break
        detections_vehicles = detect_objects(frame, coco_model, conf=0.3)
        detetions_ = []

        # Kiểm tra detections trước khi tiếp tục
        if detections_vehicles is not None and len(detections_vehicles["class_name"]) > 0:
            # print("\nKiểu dữ liệu: ", detections["class_name"])
            # print("\nLen: ", len(detections["class_name"]))
            detections = byte_tracker.update(
                detections=detections_vehicles
            )
            

            # Vẽ kết quả lên khung hình
            frame = draw_boxes(frame, detections, box_annatator, lables_annatator)

        # 6. Hiển thị khung hình
        cv2.imshow("YOLOv8 - RTMP Stream", frame)

        # Nhấn phím ESC (mã ASCII 27) để thoát khỏi cửa sổ
        if cv2.waitKey(30) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
