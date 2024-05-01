import streamlit as st
import pandas as pd
import pickle
import sklearn

# Loading the model
with open('model.pkl','rb') as file:
    model = pickle.load(file)



st.title(" Model Deployment Testing ")
with st.form("my_form"):
    st.write("Enter feature values")

    fixed_acidity = st.number_input("fixed_acidity", format="%.2f")
    volatile_acidity = st.number_input("volatile_acidity", format="%.2f")
    citric_acid = st.number_input("citric_acid", format="%.2f")
    residual_sugar = st.number_input("residual_sugar", format="%.2f")
    chlorides = st.number_input("chlorides", format="%.2f")
    free_sulfur_dioxide = st.number_input("free_sulfur_dioxide", format="%.2f")
    total_sulfur_dioxide = st.number_input("total_sulfur_dioxide", format="%.2f")
    density = st.number_input("density", format="%.2f")
    pH = st.number_input("pH", format="%.2f")
    sulphates = st.number_input("sulphates", format="%.2f")
    alcohol = st.number_input("alcohol", format="%.2f")

    submitted = st.form_submit_button("Predict")

    if submitted:
        features = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
             free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]        
        prediction = model.predict(features)
        st.write(f"Prediction : {prediction}")
