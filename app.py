from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import requests
import os

app = Flask(__name__)

# URL of the hosted model file
MODEL_URL = "https://drive.google.com/uc?export=download&id=1qUkxkPapJEohXFcImb7DFbsmLJnHn-7N"


# Local path where the model will be saved and loaded
model_path = "python.h5"

# Check if the model file exists locally; if not, download it

if not os.path.exists("python.h5"):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open("python.h5", "wb") as file:
        file.write(response.content)

# Load the model
model = load_model("python.h5")


# Normalization function
def normalize_input(features):
    """
    Normalize numerical features and encode categorical ones.
    features: numpy array of input values
    """
    feature_min = np.array([2850, 0, 0, 18, 0, 0, 0, 0]) 
    feature_max = np.array([4559, 1, 1, 77, 1, 1, 8.607, 1])  
    
    normalized_features = features.copy()
    for i in range(len(features)):
        if feature_min[i] is not None and feature_max[i] is not None: 
            normalized_features[i] = (features[i] - feature_min[i]) / (feature_max[i] - feature_min[i])
    return normalized_features

@app.route("/")
def home():
    return render_template("index.html")  # Front-end

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Received data:", request.form)  # Debug input
        data = [
            float(request.form["Ort_transformed"]),
            float(request.form["guide_transformed"]),
            float(request.form["gender_transformed"]),
            float(request.form["age_transformed"]),
            float(request.form["Slow_transformed"]),
            float(request.form["Pre_acclimatization_transformed"]),
            float(request.form["Knowledge_score_transformed"]),
            float(request.form["AMS_history"])
        ]
        print("Extracted Data:", data)  # Debug extracted data

        # Normalize and predict
        data_array = np.array(data)
        normalized_data = normalize_input(data_array)
        print("Normalized Data:", normalized_data)  # Debug normalized data

        normalized_data = normalized_data.reshape(1, -1)
        prediction = model.predict(normalized_data)
        probability = prediction[0][0]
        print("Prediction Probability:", probability)  # Debug prediction

        return jsonify({"probability": f"{probability * 100:.2f}%"})
    except Exception as e:
        print("Error occurred:", str(e))  # Debug errors
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
