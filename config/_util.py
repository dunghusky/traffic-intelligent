import string
# import easyocr
import numpy as np
from paddleocr import PaddleOCR

# Initialize the OCR reader
# reader = easyocr.Reader(["en"], gpu=False)
ocr = PaddleOCR(lang='en', rec_algorithm='CRNN')

# Mapping dictionaries for character conversion
dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5", "B": "8"}


dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S", "8": "B"}

dict_int_to_char_ = {
    "4": "A",
    "6": "G",
    "5": "S",
    "8": "B",
}

valid_characters = set("ABCDEFGHKLMNPSTUVXYZ")


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, "w") as f:
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                "frame_nmr",
                "car_id",
                "car_bbox",
                "license_plate_bbox",
                "license_plate_bbox_score",
                "license_number",
                "license_number_score",
            )
        )

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                # print(results[frame_nmr][car_id])
                if (
                    "car" in results[frame_nmr][car_id].keys()
                    and "license_plate" in results[frame_nmr][car_id].keys()
                    and "text" in results[frame_nmr][car_id]["license_plate"].keys()
                ):
                    f.write(
                        "{},{},{},{},{},{},{}\n".format(
                            frame_nmr,
                            car_id,
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["car"]["bbox"][0],
                                results[frame_nmr][car_id]["car"]["bbox"][1],
                                results[frame_nmr][car_id]["car"]["bbox"][2],
                                results[frame_nmr][car_id]["car"]["bbox"][3],
                            ),
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["license_plate"]["bbox"][0],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][1],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][2],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][3],
                            ),
                            results[frame_nmr][car_id]["license_plate"]["bbox_score"],
                            results[frame_nmr][car_id]["license_plate"]["text"],
                            results[frame_nmr][car_id]["license_plate"]["text_score"],
                        )
                    )
        f.close()


def license_complies_format_car(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 9:
        return False

    # Kiểm tra từng ký tự trong chuỗi văn bản biển số xe
    if (
        # Hai ký tự đầu phải là số hoặc ký tự có thể chuyển đổi thành số
        (text[0] in "0123456789" or text[0] in dict_char_to_int.keys())
        and (text[1] in "0123456789" or text[1] in dict_char_to_int.keys())
        # Ký tự thứ ba là chữ cái (nhóm đăng ký)
        and (text[2] in valid_characters or text[2] in dict_int_to_char_.keys())
        # Ký tự thứ tư là dấu phân cách (dấu chấm hoặc dấu "•")
        and text[3] in ["•", "-"]
        # Năm ký tự tiếp theo là số hoặc ký tự có thể chuyển đổi thành số
        and all(
            (char in "0123456789" or char in dict_char_to_int.keys())
            for char in text[4:]
        )
    ):
        return True
    else:
        return False


def format_license_car(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ""
    mapping = {
        0: dict_char_to_int,  # Ký tự đầu tiên (có thể là số hoặc ký tự chuyển đổi)
        1: dict_char_to_int,  # Ký tự thứ hai (tương tự trên)
        2: dict_int_to_char,  # Ký tự chữ cái hoặc chuyển đổi từ số
        4: dict_char_to_int,  # Năm ký tự số (hoặc ký tự có thể chuyển đổi)
        5: dict_char_to_int,
        6: dict_char_to_int,
        7: dict_char_to_int,
        8: dict_char_to_int,
    }

    for j in range(len(text)):
        # Chuyển đổi ký tự nếu có trong từ điển ánh xạ
        if j in mapping and text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate_car(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    license_plate_crop_thresh = ocr.ocr(license_plate_crop)

    # Kiểm tra nếu kết quả OCR trả về None hoặc rỗng
    if not license_plate_crop_thresh:
        print("No text detected by OCR.")
        return 0, 0

    detected_texts = []
    total_score = 0.0

    for detection_group in license_plate_crop_thresh:
        if not detection_group:
            print("Warning: Detected None in detection group.")
            continue

        for detection in detection_group:
            print("\nTest: ", detection)
            bbox, (text, score) = detection

            # Chuẩn hóa văn bản: Chuyển sang in hoa, loại bỏ khoảng trắng
            text = text.upper().replace(" ", "")

            # Nếu ký tự thứ 7 là dấu chấm (.), loại bỏ ký tự này
            if len(text) > 7 and text[7] == ".":
                text = text[:7] + text[8:]
                print("Formatted text without dot: ", text)
            else:
                print("No dot found, keeping text as is: ", text)

            # Thêm văn bản vào danh sách nếu không rỗng
            if text and score > 0.5:  # Chỉ lấy các kết quả có độ tin cậy cao hơn 0.5
                detected_texts.append(text)
                total_score += score

    if not detected_texts:
        print("No valid text detected.")
        return 0, 0

    # Ghép các phần tử thành một chuỗi duy nhất
    combined_text = (
        "-".join(detected_texts) if len(detected_texts) > 1 else detected_texts[0]
    )

    # Loại bỏ dấu chấm nếu cần thiết
    combined_text = combined_text.replace(".", "")

    print("Combined text: ", combined_text)

    if license_complies_format_car(combined_text):
        average_score = total_score / len(detected_texts)
        return format_license_car(combined_text), average_score

    return 0, 0


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """

    xyxy, _, conf_license, class_id_license, _, _ = license_plate
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
