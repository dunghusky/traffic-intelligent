from ultralytics import YOLO
import numpy as np
import cv2
import supervision as sv

detect_license_plate = YOLO("./model/train/checkpoints_lights/train/weights/best.pt")


cap = cv2.VideoCapture("./file_path/16h15.25.9.22.mp4")
def detect_traffic_light(
    traffic_lights, existing_traffic_lights=None
):  # (array([     1371.7,      274.18,      1432.8,      407.35], dtype=float32), None, 0.9041419, 0, 1, {'class_name': 'Green'})

    updated_traffic_lights = (
        existing_traffic_lights.copy() if existing_traffic_lights else {}
    )
    for traffic_light in traffic_lights:
        xyxy, _, conf_license, class_id_license, track_id, class_name = (
            traffic_light  # 0: green, 1:red, 2:yellow
        )
        class_name = class_name.get("class_name", None)
        # x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

        updated_traffic_lights[track_id] = class_name

    for track_id in existing_traffic_lights:
        if track_id not in updated_traffic_lights:
            updated_traffic_lights[track_id] = existing_traffic_lights[
                track_id
            ]  # Giữ trạng thái cũ nếu không phát hiện lại đèn
    return updated_traffic_lights


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


label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)
box_annotator = sv.BoxAnnotator(thickness=2)
ret = True
while ret:
    # frame_nmr += 1
    ret, frame = cap.read()

    if not ret:
        break

    results = detect_license_plate.track(frame, persist=True, conf=0.01)[0]
    detections = sv.Detections.from_ultralytics(results)

    detection_results = []
    labels = []

    # for xyxy, confidence, class_id, track_id, class_name in zip(
    #     detections.xyxy,
    #     detections.confidence,
    #     detections.class_id,
    #     detections.tracker_id,
    #     detections["class_name"],
    # ):
    #     x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    #     detection_results.append([x1, y1, x2, y2, class_id, track_id])
    #     print("Detections: ", detection_results)

    #     label = f"#{track_id} {class_name} {confidence:.2f}"
    #     labels.append(label)
    detection_lights = objects_tracking(frame, detect_license_plate)
    existing_traffic_lights = detect_traffic_light(detection_lights)
    print("Test: ", existing_traffic_lights)

    # annotated_frame = box_annotator.annotate(detections=detections, scene=frame)

    # annotated_frame = label_annotator.annotate(
    #     labels=labels, scene=frame, detections=detections
    # )

    cv2.imshow("OT", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
