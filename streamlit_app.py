import streamlit as st
import pickle
import numpy as np

# Load the saved model
model_filename = 'best_rf_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Title and instructions
st.title("Health Status Prediction App")
st.write("Enter the following health metrics to predict if an individual is 'healthy' or 'unhealthy'.")

# Collect user input for each feature
age = st.number_input("Age", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
cholesterol = st.number_input("Cholesterol Level", min_value=0.0)
glucose_level = st.number_input("Glucose Level", min_value=0.0)
heart_rate = st.number_input("Heart Rate", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
exercise_hours = st.number_input("Exercise Hours", min_value=0.0)
water_intake = st.number_input("Water Intake", min_value=0.0)
stress_level = st.number_input("Stress Level", min_value=0.0)
smoking = st.selectbox("Smoking", [0, 1])
alcohol = st.selectbox("Alcohol Consumption", [0, 1])
diet = st.selectbox("Diet (0 = Unhealthy, 1 = Healthy)", [0, 1])
mental_health = st.selectbox("Mental Health (0 = Poor, 1 = Good)", [0, 1])
physical_activity = st.selectbox("Physical Activity (0 = Low, 1 = High)", [0, 1])
medical_history = st.selectbox("Medical History (0 = None, 1 = Existing)", [0, 1])
allergies = st.selectbox("Allergies (0 = None, 1 = Yes)", [0, 1])
diet_type_vegan = st.checkbox("Vegan Diet")
diet_type_vegetarian = st.checkbox("Vegetarian Diet")
blood_group_ab = st.checkbox("Blood Group AB")
blood_group_b = st.checkbox("Blood Group B")
blood_group_o = st.checkbox("Blood Group O")

# Predict button
if st.button("Predict Health Status"):
    # Organize the inputs into a single array
    input_data = np.array([
        age, bmi, blood_pressure, cholesterol, glucose_level, heart_rate, sleep_hours, exercise_hours, water_intake,
        stress_level, smoking, alcohol, diet, mental_health, physical_activity, medical_history, allergies,
        diet_type_vegan, diet_type_vegetarian, blood_group_ab, blood_group_b, blood_group_o
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.write("The model predicts: Unhealthy")
    else:
        st.write("The model predicts: Healthy")
