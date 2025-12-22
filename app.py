import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from tqdm import tqdm
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Model configuration
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
model.load_state_dict(torch.load('deepfake_mode_MPS.pth', map_location=device))
model = model.to(device)
model.eval()

# Flask configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.secret_key = 'supersecretkey123'  # Change this in production

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_video_frames(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_interval == 0:
            frames.append(frame)
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    return frames

def predict_deepfake(video_path):
    frames = extract_video_frames(video_path)
    
    if not frames:
        raise ValueError("No frames extracted from video")
    
    predictions = []  

    with torch.no_grad():
        for frame in tqdm(frames, desc="Processing Frames"):
            try:
                image = transform(frame).unsqueeze(0).to(device)
                output = model(image)
                _, pred = torch.max(output, 1)
                predictions.append(pred.item())
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    if not predictions:
        raise ValueError("No predictions made - processing failed")
    
    fake_percentage = (predictions.count(1) / len(predictions)) * 100
    return 'FAKE' if fake_percentage > 50 else 'REAL'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        try:
            result = predict_deepfake(upload_path)
            return render_template('index.html', result=result, filename=filename)
        except Exception as e:
            flash(f'Error processing video: {str(e)}')
            return redirect(url_for('index'))
        finally:
            if os.path.exists(upload_path):
                os.remove(upload_path)
    else:
        flash('Allowed file types are: mp4, avi, mov')
        return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
