import os
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename 
import cv2
import numpy as np
from tensorflow.keras.models import load_model #type: ignore
from Prediction import predict_image

# Initialize Flask app
app = Flask(__name__, template_folder='.')

# Configure paths and settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'unet_model.keras')
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load model with proper error handling
try:
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check if model file exists.'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")

        # Read and preprocess image
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Failed to read image file")

        # Preprocess image
        processed_img = cv2.resize(img, (256, 256))
        processed_img = processed_img.astype('float32') / 255.0
        processed_img = np.expand_dims(processed_img, axis=0)

        # Get prediction
        prediction_image = predict_image(processed_img, model)
        if prediction_image is None:
            raise ValueError("Failed to generate prediction")

        # Save prediction
        prediction_filename = f'prediction_{filename}'
        prediction_filepath = os.path.join(app.config['UPLOAD_FOLDER'], prediction_filename)
        cv2.imwrite(prediction_filepath, prediction_image)

        # Generate URLs
        original_url = url_for('static', filename=f'uploads/{filename}')
        prediction_url = url_for('static', filename=f'uploads/prediction_{filename}')

        return jsonify({
            'success': True,
            'original_image': original_url,
            'prediction_image': prediction_url
        })

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True, port=5500)
