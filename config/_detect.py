import supervision as sv

from collections import deque

def detect_objects(frame, model, conf=0.05, iou=0.5):
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


def objects_tracking(frame, model, conf=0.1, iou=0.5, classes=None):
    """
    Nhận diện vật thể trên khung hình hiện tại.
    Args:
    - frame: Khung hình hiện tại từ video.
    - model: Mô hình YOLO đã được tải.

    Returns:
    - detections: Kết quả nhận diện.
    """
    # results = model(frame)[0]
    results = model.track(frame, persist=True, conf=conf, classes=classes)[0]
    detections = sv.Detections.from_ultralytics(results)
    return detections


def draw_boxes(
    frame, detections, labels, box_annotator, lables_annotator, polygon_zone_annotator
):

    frame = box_annotator.annotate(detections=detections, scene=frame)

    frame = lables_annotator.annotate(labels=labels, scene=frame, detections=detections)

    frame = polygon_zone_annotator.annotate(scene=frame)

    return frame


def initialize_and_annotators(polygon):

    box_annotator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)
    polygon_zone = sv.PolygonZone(polygon=polygon)
    polygon_zone_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, thickness=1)
    # line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    # line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=0, text_scale=0)
    return (box_annotator, label_annotator, polygon_zone, polygon_zone_annotator)


# def detect_traffic_light(
#     traffic_lights, existing_traffic_lights=None
# ):  # (array([     1371.7,      274.18,      1432.8,      407.35], dtype=float32), None, 0.9041419, 0, 1, {'class_name': 'Green'})
#     # Nếu đã có buffer, sử dụng nó, nếu không thì khởi tạo mới

#     labels = []
#     updated_traffic_lights = (
#         existing_traffic_lights.copy() if existing_traffic_lights else {}
#     )
#     for traffic_light in traffic_lights:
#         xyxy, _, conf_license, class_id_license, track_id, class_name = (
#             traffic_light  # 0: green, 1:red, 2:yellow
#         )
#         class_name = class_name.get("class_name", None)
#         # x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
#         label = label = f"#{track_id} {class_name} {conf_license:.2f}"
#         labels.append(label)

#         updated_traffic_lights[track_id] = class_name

#     if existing_traffic_lights:
#         for track_id in existing_traffic_lights:
#             if track_id not in updated_traffic_lights:
#                 updated_traffic_lights[track_id] = existing_traffic_lights[
#                     track_id
#                 ]  # Giữ trạng thái cũ nếu không phát hiện lại đèn
#     return updated_traffic_lights, labels


def detect_traffic_light(
    traffic_lights, existing_buffer=None, BUFFER_SIZE=30
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
