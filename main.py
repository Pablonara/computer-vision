from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, secure_filename
from ultralytics import YOLO
import os 
import request  
import torch
from PIL import Image

app = Flask(__name__)

model_path = "model/trained.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(model_path)
model.to(device)

uploadFolder = 'uploads'

app.config['uploadFolder'] = uploadFolder

@app.route('/api') # seperate api route later if needed
def api(): # fix naming
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'There is no file1 in the form!'
        file1 = request.files['file1']
       	filename = file1.filename
        filepath = os.path.join(app.config['uploadFOlder'], filename)
        file1.save(filepath)
        filename, fileExtension = os.path.splitext(filepath)
        fileExtension = set(['.png', '.jpg', '.jpeg'])
        
        if fileExtension in imageExtensions: 
            results = model(filepath, stream=False, conf=0.5) 
        # Process results list
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
#                result.show()  # display to screen
                result.save(filename=filepath)  # save to disk
                file_name = os.path.basename(filepath)
                return send_from_directory(app.config['UPLOAD_FOLDER'], file_name)
        
        else:
            #return(process_video(filepath))
            cap = cv2.VideoCapture(filepath)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
            output_filepath = os.path.splitext(filepath)[0] + '_processed.mp4'
            out = cv2.VideoWriter(output_filepath, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, stream=False, conf=0.5)
                framenum = 0 
                for result in results:
                    boxes = result.boxes  # Boxes object for bounding box outputs
                    masks = result.masks  # Masks object for segmentation masks outputs
                    keypoints = result.keypoints  # Keypoints object for pose outputs
                    probs = result.probs  # Probs object for classification outputs
                    obb = result.obb  # Oriented boxes object for OBB outputs
        #                result.show()  # display to screen        
                    result.save(filename="concatonate.png")
                    writeOutFrame = cv2.imread("concatonate.png") # cv2 doesn't work direct importing yolov8, need convert first
                    out.write(writeOutFrame) 
                    framenum += 1
                    print(framenum)
                    break # no need to continue after first frame because no more

    
        cap.release()
        out.release()
        return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(output_filepath), as_attachment=False)
        

    elif request.method == 'GET':
        return render_template('index.html')