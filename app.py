from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import os


app = Flask(__name__)
CORS(app)
# Load models and vectorizer
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
# Note: Please ensure the model and vectorizer files are in the "models" folder
try:
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    models = {
        "version-1": joblib.load(os.path.join(MODELS_DIR, "tuned_svm_model.joblib")),
        "version-2": joblib.load(os.path.join(MODELS_DIR, "tuned_logisticregression_model.joblib")),
        "version-3": joblib.load(os.path.join(MODELS_DIR, "tuned_naivebayes_model.joblib")),
    }
except FileNotFoundError as e:
    print(f"ERROR: Model or vectorizer file not found: {e}")
    models = None
    vectorizer = None


# Web UI route
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if not models or not vectorizer:
        return jsonify({"error": "Model files are not loaded correctly on the server."}), 500
    
    data = request.get_json()
    message = data.get("message", "")
    version = data.get("version", "version-1")
    if not message or version not in models:
        return jsonify({"error": "Please enter valid email text or select a valid model."}), 400
    try:
        X = vectorizer.transform([message])
        model = models[version]
        pred = model.predict(X)[0]
        # Note: 0 = spam, 1 = not spam
        result = "🚨 Spam" if pred == 0 else "✅ Not Spam"
        return jsonify({"result": result})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=7000)