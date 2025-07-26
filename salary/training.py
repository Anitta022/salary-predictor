import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_preprocess_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Handle categorical variables
    le_education = LabelEncoder()
    le_location = LabelEncoder()
    le_job_title = LabelEncoder()
    le_gender = LabelEncoder()
    
    df['Education'] = le_education.fit_transform(df['Education'])
    df['Location'] = le_location.fit_transform(df['Location'])
    df['Job_Title'] = le_job_title.fit_transform(df['Job_Title'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['Experience', 'Age']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Save encoders and scaler
    joblib.dump(le_education, 'le_education.pkl')
    joblib.dump(le_location, 'le_location.pkl')
    joblib.dump(le_job_title, 'le_job_title.pkl')
    joblib.dump(le_gender, 'le_gender.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return df

def train_model(file_path):
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    
    # Define features and target
    X = df[['Education', 'Experience', 'Location', 'Job_Title', 'Age', 'Gender']]
    y = df['Salary']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'salary_model.pkl')
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score: {test_score:.4f}")
    
    return model

if __name__ == "__main__":
    file_path = "salary_prediction_data.csv"
    model = train_model(file_path)
    print("Model training completed and saved as 'salary_model.pkl'.")