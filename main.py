from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for 
from ultralytics import YOLO
import os 
import 

app = Flask(__name__)

uploadFolder = 'uploads'

app.config['uploadFolder'] = uploadFolder

@app.route('/api') # seperate api route later if needed
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['uploadFolder'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

if request.method == 'GET':
    
        

    return render_template('index.html')