
import streamlit as st
import pandas as pd
import joblib, os

st.set_page_config(page_title="Career Recommender", layout="centered")
st.title("🎯 AI-Based Career Recommendation System")
st.write("Developed by **NARLA JYOTHI SWAROOPA** | NASSCOM FutureSkills AI Internship")

model = joblib.load("career_model.pkl")
encoder = joblib.load("label_encoder.pkl")

traits = ['O_score','C_score','E_score','A_score','N_score',
          'Numerical Aptitude','Spatial Aptitude','Perceptual Aptitude',
          'Abstract Reasoning','Verbal Reasoning']

cols = st.columns(2)
inputs = {}
for i, t in enumerate(traits):
    with cols[i % 2]:
        inputs[t] = st.slider(t, 1.0, 10.0, 5.0)

if st.button("Recommend Career"):
    df = pd.DataFrame([[inputs[t] for t in traits]], columns=traits)
    pred = model.predict(df)
    result = encoder.inverse_transform(pred)[0]
    st.success(f"💡 Recommended Career: {result}")
    st.balloons()
