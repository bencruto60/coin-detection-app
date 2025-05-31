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

# Supabase credentials (replace with your actual values)
SUPABASE_URL = 'https://tgvycqmnzhasmxomfruq.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRndnljcW1uemhhc214b21mcnVxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0ODY3MzQyNCwiZXhwIjoyMDY0MjQ5NDI0fQ.GvALKP_8Pl0cxQCmQrrNGNMkdp0qOZWMgaLeIuK02M8'  # Store securely, not hardcoded in production

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

    results = model.predict(filepath, save=True, save_txt=False)

    # Save the annotated image
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detected_{filename}")
    annotated_image = results[0].plot()
    cv2.imwrite(result_image_path, annotated_image)

    # Example: basic class detection
    try:
        coin_label = results[0].names[results[0].boxes.cls[0].item()]
    except:
        coin_label = "Unknown"

    # Map label to value
    coin_map = {
        "1_peso": ("1 Peso", 1.00),
        "5_peso": ("5 Peso", 5.00),
        "10_peso": ("10 Peso", 10.00)
        # Add more as needed
    }

    coin_name, coin_value = coin_map.get(coin_label, ("Unknown", 0.00))

    # Get user ID from frontend (Google Auth)
    user_id = request.headers.get("X-User-ID")

    # Save to Supabase
    if user_id:
        supabase.table("savings").insert({
            "user_id": user_id,
            "coin_name": coin_name,
            "coin_value": coin_value,
            "image_url": url_for('static', filename=f'uploads/{filename}', _external=True)
        }).execute()

    return jsonify({
        'original': url_for('static', filename=f'uploads/{filename}'),
        'result': url_for('static', filename=f'uploads/detected_{filename}')
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
