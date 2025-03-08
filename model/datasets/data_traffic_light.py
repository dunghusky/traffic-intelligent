from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

ROBO_KEY = os.getenv("ROBOFLOW_DATASET_LIGHTS_API_KEY")

rf = Roboflow(api_key=ROBO_KEY)
project = rf.workspace("traficlightdetection").project("v5-j3ysc-r2yns")
version = project.version(1)
dataset = version.download("yolov11")
