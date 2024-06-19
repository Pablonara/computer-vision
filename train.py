# yolov8 training script, using roboflow repository to train

from ultralytics import YOLO

from IPython.display import display, Image

import yaml


version = None #tba after done dataset

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

model = YOLO('runs/obb/train2/weights/best.pt')

# testing here

import os
import random

random_file = random.choice(os.listdir(f"{dataset.location}/test/images"))
file_name = os.path.join(f"{dataset.location}/test/images", random_file)

results = model(file_name)

print(results[0])

