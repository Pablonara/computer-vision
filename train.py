#usr/bin/env python3
# yolov8 training script, using roboflow repository to train

from ultralytics import YOLO
from IPython.display import display, Image
import yaml
import os
import random
import roboflow
from testing import testAll


version = 1 #tba after done dataset

roboflow.login()
rf = roboflow.Roboflow()

project = rf.workspace("").project("")
dataset = project.version().download("yolov8-obb") # bounding boxes

with open(f'{dataset.location}/data.yaml', 'r') as file:
    data = yaml.safe_load(file)

data['path'] = dataset.location

with open(f'{dataset.location}/data.yaml', 'w') as file:
    yaml.dump(data, file, sort_keys=False)
    
model = YOLO('yolov8n-obb.pt')

results = model.train(data=f"{dataset.location}/data.yaml", epochs=10, imgsz=640) # 10 passes at 640x640 resolution

model.save("yolov8n-obb.pt")


testAll(dataset.location, "yolov8n-obb.pt", 10) # 10 testcases
