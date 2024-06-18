# yolov8 training script, using roboflow repository to train

from ultralytics import YOLO

from IPython.display import display, Image

version = None #tba after done dataset

roboflow.login()
rf = roboflow.Roboflow()

project = rf.workspace("").project("")
dataset = project.version().download("yolov8-obb") # bounding boxes