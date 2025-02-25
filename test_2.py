from ultralytics import YOLO
import numpy as np
import cv2
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort

from config._util import get_car, read_license_plate, write_csv

# Khởi tạo deepsort
tracker = DeepSort(max_age=30) #Sau 5 lần tracking không thấy vật thể thì chúng ta xóa vật thể khỏi bộ nhớ
# Load models
coco_model = YOLO("yolo11n.pt")  # use to detect car and motorbike
detect_license_plate = YOLO("./model/train/checkpoints/train/weights/best.pt")

# Load video
cap = cv2.VideoCapture("./file_path/20221003-102700.mp4")

results = {}
tracks = []
vehicles = [2, 3, 5, 7]
conf_threshold=0.5
# Read frames
frame_nmr = -1
ret = True
colors = np.random.randint(0, 255, size=(len(vehicles), 3))

box_annotator = sv.BoxAnnotator()

while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if not ret:
        break
    if ret and frame_nmr < 10:
        # Detect vehicles
        results[frame_nmr] = {}
        detections = coco_model(frame)[0]
        detections = sv.Detections.from_ultralytics(detections)
        detections_ = []

        for detection in detections: #detection trả về dạng: (array([214.34, 575.54, 513.62, 828.84], dtype=float32), None, 0.8545715, 2, None, {'class_name': 'car'})
            # print("\nDetection: ", detection)
            bbox, _, conf, class_id, _, additional_info = detection
            # print(f"\n bbox: {bbox}, conf: {conf}, class_id: {class_id}, additional_info: {additional_info}")
            class_name = additional_info.get('class_name', None)
            x1, y1, x2, y2 = map(int, bbox)
            # print(f"\n x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

            # if vehicles is None:
            #     if conf < conf_threshold:
            #         continue
            # else:
            #     if int(class_id) not in vehicles or conf < conf_threshold:
            #         continue

            # if conf is not None and isinstance(conf, (float, int)):  # Kiểm tra conf

            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, conf])
                # print("\nDetections: ", detections_)

        print("\n\nDetections big: ", detections_)

        # tracking Deepsort
        if len(detections_) > 0:
            # Chỉ giữ lại các phần tử có đúng độ dài là 5 và không chứa giá trị None
            detections_ = np.array([d for d in detections_ if len(d) == 5 and all(isinstance(i, (int, float)) for i in d)])
            print("\nKiểm tra định dạng Detections_: ", detections_)

            # Kiểm tra kích thước của mảng detections_
            if detections_.shape[1] == 5:
                tracks = tracker.update_tracks(detections_, frame=frame)
                print("\nTracks: ", tracks)
            else:
                print("Lỗi: Detections_ không đúng định dạng (N, 5)")
        else:
            tracks = []

        # Detect license plates
        license_plates = detect_license_plate(frame)[0]
        license_plates = sv.Detections.from_ultralytics(license_plates)

        for license_plate in license_plates:
            bbox, _, conf, class_id, _, additional_info = license_plate
            x1, y1, x2, y2 = map(int, bbox)

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, tracks)

            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # Process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            # cv2.imshow('original_crop', license_plate_crop)
            # cv2.imshow("threshold", license_plate_crop_thresh)

            # cv2.waitKey(0)

            # Read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(
                license_plate_crop_thresh
            )

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                    "license_plate": {
                        "bbox": [x1, y1, x2, y2],
                        "text": license_plate_text,
                        "bbox_score": conf,
                        "text_score": license_plate_text_score,
                    },
                }

# cv2.imshow("OT", frame)
# write_csv(results, './test.csv')

# if cv2.waitKey(1) == ord("q"):
#     break

# cap.release()
# cv2.destroyAllWindows()
