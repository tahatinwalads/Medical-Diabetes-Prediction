import streamlit as st
import joblib
import numpy as np


model =joblib.load("diabetes_model.pkl")
st.set_page_config(page_title="Diabetes_Prediction", page_icon = "🩺")
st.title("🩺 Diabetes Prediction System ")

st.markdown("""
Predict whether a patient is likely to have diabetes using a trained
Machine Learning model.
""")
st.divider()

st.subheader ("Patient Information")

col1, col2 = st.columns(2)

with col1 :
   pregnancies = st.number_input("Pregnancies", min_value=0)
   glucose = st.number_input("Glucose")
   bloodpressure = st.number_input("Blood Pressure")
   skinthickness = st.number_input("SkinThickness")

with col2 :  
   insulin = st.number_input("Insulin")
   bmi = st.number_input("BMI")
   dpf = st.number_input("Diabetes Pedigree Function")
   age = st.number_input("Age", min_value=1)

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

confidence = max (probability(0))* 100

if prediction[0] == 1:
    st.error("⚠️ Patient is likely to have Diabetes.")
    st.write(f"**Confidence:** {confidence:.2f}%")
else:
    st.success("✅ Patient is not likely to have Diabetes.")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.divider()
    st.caption("Developed by Taha Tinwala | Machine Learning Project")