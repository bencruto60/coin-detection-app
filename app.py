import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model using environment variable
model_path = os.environ.get('MODEL_PATH', 'model/best.pt')
model = YOLO(model_path)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save and process image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    
    # Run detection
    results = model.predict(img_path, conf=0.5)
    detections = []
    for box in results[0].boxes:
        detections.append({
            'label': results[0].names[int(box.cls)],
            'confidence': float(box.conf)
        })
    
    return jsonify({'detections': detections})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Uses Render's $PORT or defaults to 8000
    app.run(host='0.0.0.0', port=port)