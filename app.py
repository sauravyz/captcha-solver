import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io

#  Constants and Helper Functions (that are used in  notebook) 

# Constants
img_width = 200
img_height = 50
max_length = 5 
characters = sorted(['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y'])

# The decoding function from the notebook
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    for res in results.numpy():
        res_text = "".join([characters[c] for c in res if c != -1])
        output_text.append(res_text)
    return output_text

#  process the uploaded image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    img = img.resize((img_width, img_height))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = tf.transpose(img, perm=[1, 0, 2])  # Transpose for the RNN
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img




# Load the pre-trained model
try:
    prediction_model = tf.keras.models.load_model("prediction_model.h5")
    print("--- Model loaded successfully ---")
except Exception as e:
    print(f"!!! Error loading model: {e} !!!")
    prediction_model = None


#  Create the Flask App 

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Captcha Solver API</h1><p>Send a POST request to /predict with an image file.</p>"

@app.route("/predict", methods=["POST"])
def predict():
    if not prediction_model:
        return jsonify({"error": "Model is not loaded."}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in the 'image' field."}), 400

    try:
        file = request.files['image']
        image_bytes = file.read()

        # Process the image and make a prediction
        preprocessed_img = preprocess_image(image_bytes)
        preds = prediction_model.predict(preprocessed_img)
        predicted_text = decode_batch_predictions(preds)[0]

        return jsonify({"predicted_text": predicted_text})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    
    app.run(host="0.0.0.0", port=8080)