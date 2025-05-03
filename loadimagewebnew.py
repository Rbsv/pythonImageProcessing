from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Dictionary to store the models for each device
device_models = {}

# Function to load the model dynamically based on device name
def load_device_model(device_name):
    if device_name not in device_models:
        try:
            model_path = f'{device_name}_model.h5'
            print(f"Loading model from {model_path}")
            model = load_model(model_path)
            device_models[device_name] = model  # Cache the model
            print(f"Model loaded for {device_name}")
        except Exception as e:
            print(f"Error loading model for {device_name}: {str(e)}")
            return None
    return device_models[device_name]

# Function to process and predict the result
def process_image(file, device_name):
    # Load the model based on device name
    model = load_device_model(device_name)
    if not model:
        return None, "Model not found"

    try:
        # Convert the image file to PIL Image
        img = Image.open(file)
        img = img.resize((224, 224))  # Resize image to match model input size
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the loaded model
        prediction = model.predict(img_array)

        # Assuming a binary classification (0: not ok, 1: ok)
        status = "ok" if prediction[0] > 0.5 else "not ok"
        return status, None

    except Exception as e:
        return None, f"Error processing image: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if 'file' and 'device_name' are in the request
    if 'file' not in request.files or 'device_name' not in request.form:
        return jsonify({"error": "Missing image file or device name"}), 400

    file = request.files['file']
    device_name = request.form['device_name']

    # Process the image and get the result
    result, error = process_image(file, device_name)
    if error:
        return jsonify({"error": error}), 500

    return jsonify({"device_name": device_name, "status": result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
