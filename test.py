import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
# from paddleocr import PaddleOCR
from config import _util
# import easyocr
# Initialize the OCR reader
# reader = easyocr.Reader(["en"], gpu=False)
# ocr = PaddleOCR(lang="en", rec_algorithm="CRNN")

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


# def deskew_image(image, angle_threshold=7, max_rotation=7):
#     """
#     Cân chỉnh ảnh nếu góc nghiêng lớn hơn một ngưỡng nhất định,
#     nhưng luôn xoay phải (cùng chiều kim đồng hồ) và giới hạn góc xoay tối đa 5 độ.

#     Args:
#         image (numpy.ndarray): Ảnh đầu vào (grayscale hoặc binary).
#         angle_threshold (int): Ngưỡng để quyết định có xoay ảnh hay không.
#         max_rotation (int): Giới hạn góc xoay tối đa (mặc định 5 độ).

#     Returns:
#         numpy.ndarray: Ảnh đã được xoay hoặc giữ nguyên nếu không cần thiết.
#     """
#     coords = np.column_stack(np.where(image > 0))  # Lấy tọa độ của pixel có giá trị
#     rect = cv2.minAreaRect(coords)  # Lấy hình chữ nhật nhỏ nhất chứa toàn bộ chữ
#     angle = rect[-1]  # Lấy góc nghiêng của ảnh

#     # Chỉnh lại góc về khoảng hợp lý
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle

#     # Nếu góc nhỏ hơn ngưỡng `angle_threshold`, giữ nguyên ảnh
#     if abs(angle) < angle_threshold:
#         print(f"⚡ Ảnh đã thẳng (góc {angle:.2f} độ), không cần xoay.")
#         return image

#     # Giới hạn góc xoay tối đa và đảm bảo luôn xoay phải (cùng chiều kim đồng hồ)
#     if angle > 0:
#         angle = min(angle, max_rotation)  # Xoay phải tối đa 5 độ
#     else:
#         angle = -min(abs(angle), max_rotation)  # Giữ hướng xoay phải

#     print(f"🔄 Xoay ảnh {angle:.2f} độ theo chiều kim đồng hồ.")

#     # Xoay ảnh theo chiều kim đồng hồ
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, abs(angle), 1.0)  # Xoay theo chiều kim đồng hồ
#     rotated = cv2.warpAffine(
#         image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
#     )

#     return rotated


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
        # print("\nlicense_plate_crop: ", license_plate_crop)

        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        # print("\nlicense_plate_crop: ", license_plate_crop_gray)
        # license_plate_crop_gray = deskew_image(license_plate_crop_gray)
        # # Lưu ảnh biển số xe đã cắt vào folder
        file_name = f"frame_plate_{license_plate_index}.png"
        save_path = os.path.join(output_folder, file_name)
        cv2.imwrite(save_path, license_plate_crop_gray)
        print(f"Saved cropped license plate: {save_path}")

        license_plate_text, license_plate_text_score = _util.read_license_plate_motobike(
            license_plate_crop_gray
        )
        print("license_plate_crop_thresh: ", license_plate_text)
        print("\nlicense_plate_text_score: ", license_plate_text_score)

    return license_plate_text, license_plate_text_score

# Chạy chương trình
if __name__ == "__main__":
    detect_image("./file_path/cropped_license_plates_1/frame_24_plate_158.png")
