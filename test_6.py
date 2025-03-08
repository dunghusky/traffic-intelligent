import cv2
import numpy as np

def select_polygon_points(video_path):
    def click_event(event, x, y, flags, param):
        """Hàm xử lý sự kiện chuột để lấy tọa độ điểm."""
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Điểm đã chọn: ({x}, {y})")
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Vẽ điểm trên ảnh
            cv2.imshow("Frame", frame)

    # Đọc video
    video_path = video_path  # Đường dẫn tới video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Không thể đọc video")
        exit()

    points = []  # Danh sách lưu tọa độ
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", click_event)

    print("Nhấp vào các điểm trên hình để lấy tọa độ. Nhấn phím bất kỳ để thoát.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Xuất ra dạng polygon
    polygon = np.array(points)
    print("Tọa độ cho PolygonZone:")
    print(polygon)
    return polygon
