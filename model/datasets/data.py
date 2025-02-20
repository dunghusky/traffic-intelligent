from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

ROBO_KEY = os.getenv("ROBOFLOW_DATASET_API_KEY")

rf = Roboflow(api_key=ROBO_KEY)
project = rf.workspace("projectgraduation").project("vehicle-registration-plates-trudk-uvsbu")
version = project.version(2)
dataset = version.download("yolov11")

