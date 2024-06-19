import random
from IPython.display import display, Image
import yaml
import os
from ultralytics import YOLO
import fnmatch
from pathlib import Path

def testAll(datasetLocation, modellocation, numTests):
    model = YOLO(modellocation)

    # images = Path(datasetLocation).glob('*.jpg')
    # numberOfTestcases = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
    # testing here
    for i in range(numTests):
        detections = results.obb.boxes() # No one documented this in documentation sob, you need .obb for obb boxes
        randomFile = random.choice(os.listdir(f"{datasetLocation}/test/images"))
        fileName = os.path.join(f"{datasetLocation}/test/images", randomFile)
        results = model(fileName)
        print(results[0])
        if detections > 0:
            # print("Testcase " + str(i) + " sucess") 
            print("Detections: " + str(detections))