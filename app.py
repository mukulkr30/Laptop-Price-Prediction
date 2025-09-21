from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("Laptop_price_model.pkl")
scalar = joblib.load("Scaling.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ram_size = float(data['ramsize'])
        storage_size = float(data['storagesize'])
        processor_speed = float(data['processor'])
        screen_size = float(data['screensize'])
        brand = data['brand']


        # Map brand to one-hot encoding
        brand_mapping = {
            'asus': [1, 0, 0, 0, 0],
            'acer': [0, 1, 0, 0, 0],
            'dell': [0, 0, 1, 0, 0],
            'hp': [0, 0, 0, 1, 0],
            'lenovo': [0, 0, 0, 0, 1]
        }

        # Prepare input for prediction
        features = np.array([[ram_size, storage_size, processor_speed, screen_size]])
        scaled_features = scalar.transform(features)
        brand_encoded = np.array([brand_mapping[brand]])

        # Combine features
        final_input = np.hstack([scaled_features, brand_encoded])

        # Make prediction
        prediction = model.predict(final_input)

        # Apply the adjustment factor (1.055) from your training code
        adjusted_prediction = prediction[0] * 1.055

        # Return JSON response instead of rendering template
        return jsonify({'prediction': adjusted_prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)