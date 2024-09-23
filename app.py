from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# List of class labels
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Preprocess input image function
def preprocess_image(image):
    image = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    return input_arr

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        image_path = os.path.join('uploads', image_file.filename)
        image_file.save(image_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)

        # Perform prediction
        predictions = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index]

        # Return the prediction as JSON
        return jsonify({
            'predicted_class': predicted_class_label,
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
