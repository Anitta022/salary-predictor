import pandas as pd
import joblib

def predict_salary_with_input():
    # Load encoders and scaler
    le_education = joblib.load('le_education.pkl')
    le_location = joblib.load('le_location.pkl')
    le_job_title = joblib.load('le_job_title.pkl')
    le_gender = joblib.load('le_gender.pkl')
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('salary_model.pkl')
    
    # Display valid options for categorical inputs
    print("\nValid options for input:")
    print(f"Education: {list(le_education.classes_)}")
    print(f"Location: {list(le_location.classes_)}")
    print(f"Job Title: {list(le_job_title.classes_)}")
    print(f"Gender: {list(le_gender.classes_)}")
    
    # Collect user inputs
    while True:
        try:
            education = input("\nEnter Education (e.g., High School, Bachelor, Master, PhD): ").strip()
            education_encoded = le_education.transform([education])[0]
            break
        except ValueError:
            print(f"Invalid Education. Please choose from: {le_education.classes_}")
    
    while True:
        try:
            experience = float(input("Enter Experience (years, e.g., 10): "))
            if experience < 0:
                print("Experience cannot be negative.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for Experience.")
    
    while True:
        try:
            location = input("Enter Location (e.g., Urban, Suburban, Rural): ").strip()
            location_encoded = le_location.transform([location])[0]
            break
        except ValueError:
            print(f"Invalid Location. Please choose from: {le_location.classes_}")
    
    while True:
        try:
            job_title = input("Enter Job Title (e.g., Analyst, Manager, Director, Engineer): ").strip()
            job_title_encoded = le_job_title.transform([job_title])[0]
            break
        except ValueError:
            print(f"Invalid Job Title. Please choose from: {le_job_title.classes_}")
    
    while True:
        try:
            age = float(input("Enter Age (e.g., 45): "))
            if age < 18:
                print("Age must be 18 or older.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for Age.")
    
    while True:
        try:
            gender = input("Enter Gender (e.g., Male, Female): ").strip()
            gender_encoded = le_gender.transform([gender])[0]
            break
        except ValueError:
            print(f"Invalid Gender. Please choose from: {le_gender.classes_}")
    
    # Scale numerical features using a DataFrame
    numerical_data = pd.DataFrame([[experience, age]], columns=['Experience', 'Age'])
    numerical_data_scaled = scaler.transform(numerical_data)
    experience_scaled, age_scaled = numerical_data_scaled[0]
    
    # Create feature DataFrame
    features = pd.DataFrame({
        'Education': [education_encoded],
        'Experience': [experience_scaled],
        'Location': [location_encoded],
        'Job_Title': [job_title_encoded],
        'Age': [age_scaled],
        'Gender': [gender_encoded]
    })
    
    # Make prediction
    prediction = model.predict(features)[0]
    print(f"\nPredicted Salary: ${prediction:.2f}")
    return prediction

if __name__ == "__main__":
    while True:
        predict_salary_with_input()
        continue_pred = input("\nWould you like to make another prediction? (yes/no): ").strip().lower()
        if continue_pred != 'yes':
            break
    print("Thank you for using the salary prediction tool!")