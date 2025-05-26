from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import regularizers
from PIL import Image
import io
import os

import base64
from flask_cors import CORS


def load_trained_model():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,  # Exclude the top layer as we will add our own classifier
        weights='imagenet'  # Load pre-trained weights from ImageNet
    )

    base_model.trainable = False  # Freeze base model layers for transfer learning

    # Model architecture
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs)  # Apply MobileNetV2 as the base feature extractor
    x = layers.GlobalAveragePooling2D()(x)  # Global average pooling to reduce the output size
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.002))(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    

    # Get the path to the current file's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Join with the model filename
    MODEL_PATH = os.path.join(BASE_DIR, 'MobileNetV2_Full_weight.h5')

    # Load model weights
    model.load_weights(MODEL_PATH)

    return model


def predict_image(img, model):
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return prediction

model = load_trained_model()  # Load the trained model
app = Flask(__name__)  
CORS(app, origins="*", methods=["OPTIONS", "POST", "PATCH", "GET", "PUT", "DELETE"], supports_credentials=True)

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # ensure 3 channels

        prediction = predict_image(img, model)
        print(prediction[0])
        return jsonify({
            'prediction': float(prediction[0][0]),  # Assuming binary classification
            'message': 'Prediction successful'
        })

    except Exception as e:
        return jsonify({'error': 'Invalid image', 'details': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
       