from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
from supabase import create_client
import os
import cv2
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize YOLO model
model = YOLO('model/best (1).pt')

# Debug: Print model class names
print("Model class names:", model.names)  # Add this to verify your model's classes

# Supabase credentials
SUPABASE_URL = 'https://tgvycqmnzhasmxomfruq.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRndnljcW1uemhhc214b21mcnVxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0ODY3MzQyNCwiZXhwIjoyMDY0MjQ5NDI0fQ.GvALKP_8Pl0cxQCmQrrNGNMkdp0qOZWMgaLeIuK02M8'
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run prediction with higher confidence threshold
    results = model.predict(filepath, conf=0.5)  # Increased confidence threshold

    # Save annotated image
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detected_{filename}")
    annotated_image = results[0].plot()
    cv2.imwrite(result_image_path, annotated_image)

    # Get all detections
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            label = results[0].names[class_id]
            detections.append({
                'label': label,
                'confidence': confidence
            })
            print(f"Detected: {label} (confidence: {confidence:.2f})")  # Debug output

    # Coin mapping - MUST match model's class names exactly
    coin_map = {
        "1_Back": ("1 Peso", 1.00),
        "1_Front": ("1 Peso", 1.00),
        "5 Back": ("5 Peso", 5.00),
        "5 Front": ("5 Peso", 5.00),
        "10 Back": ("10 Peso", 10.00),
        "10 Front": ("10 Peso", 10.00),
        "20 Back": ("20 Peso", 20.00),
        "20 Front": ("20 Peso", 20.00),
    }

    user_id = request.headers.get("X-User-ID")
    if user_id:
        rows_to_insert = []
        for detection in detections:
            coin_name, coin_value = coin_map.get(detection['label'], ("Unknown", 0.00))
            if coin_name != "Unknown":
                rows_to_insert.append({
                    "user_id": user_id,
                    "coin_name": coin_name,
                    "coin_value": coin_value,
                    "image_url": url_for('static', filename=f'uploads/{filename}', _external=True)
                })

        if rows_to_insert:
            try:
                response = supabase.table("savings").insert(rows_to_insert).execute()
                print("Supabase insert response:", response)
            except Exception as e:
                print("Supabase error:", e)

    # Return info on all detected coins, not just one
    return jsonify({
        'original': url_for('static', filename=f'uploads/{filename}'),
        'result': url_for('static', filename=f'uploads/detected_{filename}'),
        'detections': detections,  # send all detected coins info back
        'debug': {  # Added debug info
            'model_classes': model.names
        }
    })

# New endpoint: total value of coins per user
@app.route('/total', methods=['GET'])
def total_value():
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({'error': 'User ID missing'}), 400

    try:
        response = supabase.table("savings").select("coin_value").eq("user_id", user_id).execute()
        if response.error:
            return jsonify({'error': response.error.message}), 500

        total = sum(record['coin_value'] for record in response.data)
        return jsonify({'total_value': total})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Updated history endpoint with fixed order column and total sum in Python
@app.route('/history', methods=['GET'])
def history():
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400

    try:
        # Use detected_at (not created_at) to order because your insert uses detected_at
        response = supabase.table("savings").select("*").eq("user_id", user_id).order("detected_at", desc=True).execute()
        entries = response.data or []

        # Calculate total coin_value in Python
        total_value = sum(entry.get('coin_value', 0) for entry in entries)

        return jsonify({
            'history': entries,
            'total_value': total_value
        })
    except Exception as e:
        print("History route error:", e)
        return jsonify({'error': 'Failed to fetch history'}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
