from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))

# @app.route("/")
# def home():
#     return "API Running"

@app.route("/")
def ui():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    base_features = [
        int(data["age"]),
        int(data["sex"]),
        float(data["bmi"]),
        int(data["children"]),
        int(data["smoker"]),
        int(data["region"])
    ]

    companies = {
        "LIC": 0,
        "HDFC": 1,
        "Star Health": 2
    }

    results = {}

    for name, code in companies.items():
        features = [base_features + [code]]
        price = model.predict(features)[0]
        results[name] = round(price, 2)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
