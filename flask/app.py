import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import tensorflow as tf
from PIL import Image
import logging
from werkzeug.utils import secure_filename 
from google.cloud import storage 

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_change_me_in_prod') 

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", None)
GCS_MODEL_BLOB = "best_model.keras"
LOCAL_MODEL_DIR = "/app/model"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, GCS_MODEL_BLOB)
CLASSES_PATH = 'classes.txt'
IMG_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel('INFO')

def download_model_from_gcs(bucket_name, blob_name, destination_path):
    if not bucket_name:
         app.logger.error("GCS_BUCKET_NAME environment variable not set. Cannot download model.")
         return False
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            app.logger.error(f"Model blob gs://{bucket_name}/{blob_name} does not exist.")
            return False

        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        app.logger.info(f"Downloading model from gs://{bucket_name}/{blob_name} to {destination_path}...")
        blob.download_to_filename(destination_path)
        app.logger.info(f"Model downloaded successfully to {destination_path}.")
        return True
    except Exception as e:
        app.logger.error(f"Failed to download model from GCS: {e}", exc_info=True)
        return False

model = None
classes = []
num_classes = 0

try:
    app.logger.info(f"Loading classes from {CLASSES_PATH}...")
    with open(CLASSES_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    num_classes = len(classes)
    app.logger.info(f"Classes loaded: {classes}")
except Exception as e:
    app.logger.error(f"FATAL: Error loading classes file {CLASSES_PATH}: {e}", exc_info=True)

model_downloaded = download_model_from_gcs(GCS_BUCKET_NAME, GCS_MODEL_BLOB, LOCAL_MODEL_PATH)

if model_downloaded and os.path.exists(LOCAL_MODEL_PATH):
    try:
        app.logger.info(f"Loading model from local path {LOCAL_MODEL_PATH}...")
        model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
        app.logger.info("Model loaded successfully.")
    except Exception as e:
        app.logger.error(f"FATAL: Error loading model from {LOCAL_MODEL_PATH} after download: {e}", exc_info=True)
else:
     app.logger.error("FATAL: Model not loaded because download failed or file doesn't exist.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        app.logger.error(f"Error preprocessing image: {e}", exc_info=True)
        return None

def perform_prediction(image_bytes):
     if model is None:
         app.logger.error("perform_prediction called but model is not loaded.")
         return None 
         
     processed_image = preprocess_image(image_bytes)
     if processed_image is None:
         return None

     try:
         app.logger.info("Making prediction...")
         predictions_raw = model.predict(processed_image)
         app.logger.info("Prediction complete.")

         predicted_class_index = np.argmax(predictions_raw[0])
         if not classes or predicted_class_index >= len(classes):
              app.logger.error(f"Invalid predicted class index: {predicted_class_index}")
              return None
         predicted_class_name = classes[predicted_class_index]
         confidence = float(predictions_raw[0][predicted_class_index])
         probabilities = {classes[i]: float(predictions_raw[0][i]) for i in range(num_classes)}

         return {
             "predicted_class": predicted_class_name,
             "confidence": confidence,
             "probabilities": probabilities,
             "raw_prediction": predictions_raw[0].tolist()
         }
     except Exception as e:
         app.logger.error(f"Error during model prediction: {e}", exc_info=True)
         return None

@app.route('/', methods=['GET'])
def index():
    model_ready = model is not None and bool(classes)
    return render_template('index.html', prediction=None, model_ready=model_ready)

@app.route('/predict_form', methods=['POST'])
def predict_form():
    model_ready = model is not None and bool(classes)
    if not model_ready:
         flash('Model is not loaded, cannot process request.', 'error')
         return redirect(url_for('index'))
         
    if 'image_file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('index'))
        
    file = request.files['image_file']
    
    if file.filename == '':
        flash('No selected file.', 'warning')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read()
            prediction_result = perform_prediction(image_bytes)

            if prediction_result:
                 encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                 flash('Prediction successful!', 'success')
                 return render_template('index.html', 
                                        prediction=prediction_result, 
                                        image_data=encoded_image,
                                        model_ready=model_ready)
            else:
                 if model is None:
                     flash('Model could not be loaded. Cannot predict.', 'error')
                 else:
                     flash('Prediction failed during processing.', 'error')
                 return redirect(url_for('index'))

        except Exception as e:
            app.logger.error(f"Error handling file upload: {e}", exc_info=True)
            flash('An error occurred processing the file.', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg.', 'error')
        return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict_json_api():
    if model is None or not classes:
         return jsonify({"error": "Model or classes not loaded"}), 503

    if not request.is_json:
         return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No 'image' key found in JSON payload"}), 400

    try:
        image_data_base64 = data['image']
        image_bytes = base64.b64decode(image_data_base64)
        prediction_result = perform_prediction(image_bytes)

        if prediction_result:
             return jsonify(prediction_result)
        else:
             if model is None:
                return jsonify({"error": "Model could not be loaded. Cannot predict."}), 503
             else:
                return jsonify({"error": "Prediction failed during processing."}), 500

    except (base64.binascii.Error, ValueError):
         return jsonify({"error": "Invalid base64 image data"}), 400
    except Exception as e:
        app.logger.error(f"An internal error occurred during JSON prediction: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during prediction."}), 500

@app.route('/predict_sample/<filename>', methods=['GET'])
def predict_sample(filename):
    model_ready = model is not None and classes
    if not model_ready:
        flash('Model is not loaded, cannot process request.', 'error')
        return redirect(url_for('index'))

    safe_filename = secure_filename(filename)
    sample_image_path = os.path.join(app.static_folder, 'samples', safe_filename)

    if not os.path.exists(sample_image_path):
        app.logger.error(f"Sample image not found: {sample_image_path}")
        flash('Selected sample image not found.', 'error')
        return redirect(url_for('index'))

    try:
        with open(sample_image_path, 'rb') as f:
            image_bytes = f.read()

        prediction_result = perform_prediction(image_bytes)

        if prediction_result:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            flash('Prediction successful for sample image!', 'success')
            return render_template('index.html',
                                   prediction=prediction_result,
                                   image_data=encoded_image,
                                   model_ready=model_ready)
        else:
            flash('Prediction failed during processing for sample image.', 'error')
            return redirect(url_for('index'))

    except Exception as e:
        app.logger.error(f"Error handling sample image {safe_filename}: {e}", exc_info=True)
        flash('An error occurred processing the sample image.', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    local_debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=local_debug)
