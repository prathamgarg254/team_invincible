from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from flask_cors import CORS
import os
import numpy as np
import io
import base64
from PIL import Image, ImageChops, ImageEnhance

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to convert image to ELA
def convert_to_ela_image(image, quality):
    original_image = image.convert('RGB')

    # Resaving input image at the desired quality
    resaved_file_name = 'resaved_image.jpg'  # Predefined filename for resaved image
    original_image.save(resaved_file_name, 'JPEG', quality=quality)
    resaved_image = Image.open(resaved_file_name)

    # Pixel difference between original and resaved image
    ela_image = ImageChops.difference(original_image, resaved_image)

    # Scaling factors are calculated from pixel extremas
    extrema = ela_image.getextrema()
    max_difference = max([pix[1] for pix in extrema if pix[1] is not None])
    scale = 350.0 / max_difference if max_difference != 0 else 1

    # Enhancing ELA image to brighten the pixels
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

# Prepare image for the model
def prepare_image(image_path):
    image_size = (128, 128)
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

# Load the pre-trained Keras model
MODEL_PATH = 'model-final.h5'
model = load_model(MODEL_PATH)

# Define class labels
class_names = ['Forged', 'Authentic']

@app.route('/')
def index():
    return "Hello, this is the Flask API!"  # Basic response
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Open the image file
    original_image = Image.open(file.stream)

    # Preprocess the image for model prediction
    test_image = prepare_image(original_image)
    test_image = test_image.reshape(-1, 128, 128, 3)

    # Make a prediction
    y_pred = model.predict(test_image)
    y_pred_class = int(y_pred[0][0] > 0.5)  # Binary classification: 0 or 1
    confidence = y_pred[0][0] * 100 if y_pred[0][0] > 0.5 else (1 - y_pred[0][0]) * 100

    # Determine which image to return based on prediction
    if y_pred_class == 0:  # Forged
        ela_image = convert_to_ela_image(original_image, 90)
        buffered = io.BytesIO()
        ela_image.save(buffered, format="PNG")
        ela_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = {
            'prediction': class_names[y_pred_class],
            'confidence': f'{confidence:.2f}%',
            'ela_image_url': f'data:image/png;base64,{ela_image_base64}'  # Base64-encoded ELA image
        }
    else:  # Authentic
        buffered = io.BytesIO()
        original_image.save(buffered, format="PNG")
        uploaded_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = {
            'prediction': class_names[y_pred_class],
            'confidence': f'{confidence:.2f}%',
            'uploaded_image_url': f'data:image/png;base64,{uploaded_image_base64}'  # Base64-encoded original image
        }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode
