from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load encoders and model
le_education = joblib.load("le_education.pkl")
le_location = joblib.load("le_location.pkl")
le_job_title = joblib.load("le_job_title.pkl")
le_gender = joblib.load("le_gender.pkl")
scale = joblib.load("scaler.pkl")
model = joblib.load("salary_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    education = request.form["education"]
    experience = int(request.form["experience"])
    job_title = request.form["job_title"]
    location = request.form["location"]
    age = int(request.form["age"])
    gender = request.form["gender"]

    # Encode categorical features
    education_encoded = le_education.transform([education])[0]
    job_title_encoded = le_job_title.transform([job_title])[0]
    location_encoded = le_location.transform([location])[0]
    gender_encoded = le_gender.transform([gender])[0]

    # Combine all inputs
    input_data = np.array([[education_encoded, experience, location_encoded,
                            job_title_encoded, age, gender_encoded]])

    # Scale features
    scaled_input = scale.transform(input_data)

    # Predict
    predicted_salary = model.predict(scaled_input)[0]

    return f"<h2>Predicted Salary: ₹{int(predicted_salary):,}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
