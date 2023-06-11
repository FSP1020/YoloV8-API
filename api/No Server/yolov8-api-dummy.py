from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

import numpy as np
from PIL import Image
import base64
import io

# Create the Flask application
app = Flask(__name__)

# Define a basic route
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api', methods=['POST'])
def TEST_API():
    data = request.form.get("value")

    return jsonify(data)

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    img = Image.open(image)
    img_array = np.array(img)
    
    # Process the image array as needed
    
    # Convert the image array to RGB and save as JPEG
    img_pil = Image.fromarray(img_array)
    img_pil = img_pil.convert('RGB')
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({'image': img_base64})

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)