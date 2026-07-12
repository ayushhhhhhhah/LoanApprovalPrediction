from flask import Flask, render_template ,request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/loan_approval_model.pkl")
scaler = joblib.load("models/scaler.pkl")
required_fields = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()
        for field in required_fields:
            if field not in data:
                return {
                    "error": f"Missing field: {field}"
                }, 400

        prediction_data = pd.DataFrame([data])
        prediction_scaled = scaler.transform(prediction_data)
        prediction = model.predict(prediction_data)
        probability = model.predict_proba(prediction_data)

        print(probability)

        confidence = probability[0][prediction[0]] * 100
        confidence = round(confidence, 2)
        if prediction[0] == 1:
            result = "Loan Approved"
        else:
            result = "Loan Rejected"
        return {
            "prediction": result,
            "confidence": confidence
        }
    except Exception as e:
        return {
            "error": "Prediction failed",
            "message": str(e)
        }, 500

if __name__ == "__main__":
    app.run(debug=True)