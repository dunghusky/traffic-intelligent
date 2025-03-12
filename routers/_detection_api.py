import asyncio
from datetime import datetime
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, APIRouter
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
import uvicorn
from config import _constants, _create_file

router = APIRouter(
    prefix="/api/v1/stream",
    tags=["stream"],
)

received_labels = []
# Global storage for video URL
video_url_storage = None


# ----------------------------------------------------------#
# size : 800x600
# @router.get("/video_feed")
# def video_feed():
#     try:
#         stream_url = "rtmp://35.91.130.206:6606/live"  # "rtmp://52.88.216.148:12566/live"  # "rtmp://45.90.223.138:12586/live"  # Thay bằng stream URL thực tế: https://9500-116-105-216-200.ngrok-free.app/1
#         return StreamingResponse(
#             _stream_detect.generate_stream(stream_url),
#             media_type="multipart/x-mixed-replace; boundary=frame",
#             status_code=200,
#         )
#     except Exception as e:
#         return JSONResponse({"status": 500, "message": "Lỗi hệ thống!"})


@router.post("/send-label")
async def send_label(waste_label: WasteLabel):
    """
    API nhận nhãn mới và chèn vào đầu danh sách
    """
    # Chèn nhãn mới vào đầu danh sách
    received_labels.insert(0, waste_label.label)

    # Log để kiểm tra
    print(f"Received label: {waste_label.label}")
    print(f"Updated received_labels: {received_labels}")

    return {"message": "Label received successfully"}


@router.get("/get_video_url")
def get_video_url():
    try:
        global video_url_storage
        if video_url_storage:
            return JSONResponse(
                content={
                    "status": 200,
                    "message": "Lấy URL video thành công.",
                    "video_url": video_url_storage,
                },
                status_code=200,
            )
        else:
            return JSONResponse(
                {"status": 404, "message": "Không tìm thấy video để xem."}
            )
    except Exception as e:
        return JSONResponse({"status": 500, "message": f"Lỗi hệ thống: {str(e)}"})


@router.get("/view-labels")
async def view_labels():
    return {"received_labels": received_labels}
