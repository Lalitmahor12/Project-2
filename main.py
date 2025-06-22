# Personalized Healthcare Treatment Recommendation Engine

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Sample synthetic dataset generation (can be replaced with real data)
data = {
    'age': [45, 55, 35, 60, 40],
    'bmi': [28.0, 32.1, 24.5, 29.3, 27.8],
    'blood_pressure': [130, 150, 120, 140, 135],
    'glucose_level': [150, 180, 130, 160, 145],
    'activity_level': [2, 1, 3, 1, 2],  # 1: Low, 2: Medium, 3: High
    'genetic_marker': [1, 0, 1, 1, 0],  # 1: Risk present, 0: Not present
    'treatment_plan': [0, 1, 0, 1, 0]  # 0: Plan A, 1: Plan B
}

df = pd.DataFrame(data)

# Split features and target
X = df.drop('treatment_plan', axis=1)
y = df['treatment_plan']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, predictions))

# Function to make a personalized recommendation
def recommend_treatment(age, bmi, bp, glucose, activity, genetic):
    input_data = pd.DataFrame([[age, bmi, bp, glucose, activity, genetic]],
                              columns=['age', 'bmi', 'blood_pressure', 'glucose_level', 'activity_level', 'genetic_marker'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    plan = "Plan B (aggressive treatment)" if prediction == 1 else "Plan A (standard treatment)"
    return plan

# Example usage
example = recommend_treatment(50, 30.2, 145, 170, 1, 1)
print(f"\nRecommended Treatment Plan: {example}")
