from ultralytics import YOLO
import numpy as np
import cv2
import supervision as sv

detect_license_plate = YOLO("./model/train/checkpoints/train/weights/best.pt")

cap = cv2.VideoCapture("./file_path/16h15.5.9.22.mp4")

label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=1)
box_annotator = sv.BoxAnnotator(thickness=2)
ret = True

while ret:
    # frame_nmr += 1
    ret, frame = cap.read()

    if not ret:
        break

    results = detect_license_plate.predict(frame, conf=0.1)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections_ = []

    labels = [
        f"#{conf:.2f} {class_name} "  # {confidence:.2f}
        for conf, class_name in zip(  # , confidence
            detections.confidence, detections["class_name"]  #
        )
    ]

    frame = box_annotator.annotate(detections=detections, scene=frame)

    frame = label_annotator.annotate(labels=labels, scene=frame, detections=detections)

    cv2.imshow("OT", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
