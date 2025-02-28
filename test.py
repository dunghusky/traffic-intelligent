import os
import cv2
from ultralytics import YOLO
import supervision as sv
from paddleocr import PaddleOCR
from config import _util

# Initialize the OCR reader
# reader = easyocr.Reader(["en"], gpu=False)
ocr = PaddleOCR(lang="en", rec_algorithm="CRNN")

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


def detect_image(img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {img_path}")
    output_folder = "./file_path/cropped_license_plates"
    detect_license_plate = YOLO("./model/train/checkpoints/train/weights/best.pt")

    license_plates = detect_objects(frame, detect_license_plate, conf=0.1, iou=0.5)

    for license_plate_index, license_plate in enumerate(license_plates):
        xyxy_car, _, conf_license, class_id_license, _, _ = license_plate
        x1_license, y1_license, x2_license, y2_license = (
            xyxy_car[0],
            xyxy_car[1],
            xyxy_car[2],
            xyxy_car[3],
        )

        # Crop license plate
        license_plate_crop = frame[
            int(y1_license) : int(y2_license), int(x1_license) : int(x2_license), :
        ]

        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        # # Lưu ảnh biển số xe đã cắt vào folder
        file_name = f"frame_plate_{license_plate_index}.png"
        save_path = os.path.join(output_folder, file_name)
        cv2.imwrite(save_path, license_plate_crop)
        print(f"Saved cropped license plate: {save_path}")

        license_plate_text, license_plate_text_score = _util.read_license_plate_car(
            license_plate_crop_gray
        )
        print("license_plate_crop_thresh: ", license_plate_text)
        print("\license_plate_text_score: ", license_plate_text_score)

    return license_plate_text, license_plate_text_score

# Chạy chương trình
if __name__ == "__main__":
    detect_image("./file_path/frame_15_plate_0.png")
