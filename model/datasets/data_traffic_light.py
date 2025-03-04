from roboflow import Roboflow

rf = Roboflow(api_key="l6Ho8j3l6wmLGp96t3xP")
project = rf.workspace("traficlightdetection").project("trafic-light-detection-wi5z8")
version = project.version(1)
dataset = version.download("yolov11")
