from ultralytics import YOLO
import numpy as np
import cv2
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort

# Khởi tạo deepsort
tracker = DeepSort(
    max_age=30
)  # Sau 5 lần tracking không thấy vật thể thì chúng ta xóa vật thể khỏi bộ nhớ
# Load models
coco_model = YOLO("yolo11n.pt")  # use to detect car and motorbike
detect_license_plate = YOLO("./model/train/checkpoints/train/weights/best.pt")

# Load video
cap = cv2.VideoCapture("./file_path/20221003-102700.mp4")

tracks = []
vehicles = [2, 3, 5, 7]
conf_threshold = 0.5
# Read frames
# frame_nmr = -1
ret = True
colors = np.random.randint(0, 255, size=(len(vehicles), 3))

box_annotator = sv.BoxAnnotator()

while ret:
    # frame_nmr += 1
    ret, frame = cap.read()

    if not ret:
        break
    # if ret and frame_nmr < 10:
    # Detect vehicles
    results = coco_model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections_ = []

    for (
        detection
    ) in (
        detections
    ):  # detection trả về dạng: (array([     214.34,      575.54,      513.62,      828.84], dtype=float32), None, 0.8545715, 2, None, {'class_name': 'car'})
        bbox, _, conf, class_id, _, additional_info = detection
        class_name = additional_info.get("class_name", None)
        x1, y1, x2, y2 = map(int, bbox)

        if vehicles is None:
            if conf < conf_threshold:
                continue
        else:
            if int(class_id) not in vehicles or conf < conf_threshold:
                continue

        detections_.append([[x1, y1, x2 - x1, y2 - y1], conf, class_id])

        # tracking Deepsort
        tracks = tracker.update_tracks(detections_, frame=frame)

        # Draw
        for track in tracks:
            if track.is_confirmed():
                print("Track: ", track)
                track_id = track.track_id

        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)
        color = colors[class_id]
        B, R, G = map(int, color)

        label = f"{class_id}-{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, R, G), 2)
        cv2.rectangle(frame, (x1-1, y1-20), (x1+len(label)*12, y1), (B,G,R), -1)
        cv2.putText(img=frame, text=label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, org=(x1+5, y1-8), fontScale=0.5, color=(255, 255, 255), thickness=2)

        cv2.imshow("OT", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    # cap.release()
    # cv2.destroyAllWindows()
    # Detect license plates
    detections = detect_license_plate(frame)[0]
# Assign license plate to car

# Crop license plate

# Process license plate
