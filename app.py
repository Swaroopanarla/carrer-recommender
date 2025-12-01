
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Career Recommender", layout="centered")

st.title("🎯 AI-Based Career Recommendation System")
st.write("Developed by **NARLA JYOTHI SWAROOPA** | NASSCOM FutureSkills AI Internship")
st.caption("Rate your personality, technical skills, soft skills, and experience from 1 (low) to 10 (high).")

model = joblib.load("career_model.pkl")
encoder = joblib.load("label_encoder.pkl")

traits = [
    'O_score', 'C_score', 'E_score', 'A_score', 'N_score',
    'Python', 'Java', 'C_CPP',
    'Web_Development', 'Data_Analysis', 'Machine_Learning',
    'Database', 'Cloud_Computing', 'Cybersecurity',
    'UI_UX_Design', 'Business_Analysis',
    'Leadership', 'Communication', 'Problem_Solving',
    'Creativity', 'Public_Speaking', 'Teamwork',
    'Experience'
]

cols = st.columns(2)
inputs = {}

for i, t in enumerate(traits):
    label = t.replace("_", " ")
    with cols[i % 2]:
        inputs[t] = st.slider(label, 1.0, 10.0, 5.0)

if st.button("Recommend Career"):
    df = pd.DataFrame([[inputs[t] for t in traits]], columns=traits)
    pred = model.predict(df)
    result = encoder.inverse_transform(pred)[0]
    st.success(f"💡 Recommended Career: {result}")
    st.balloons()
