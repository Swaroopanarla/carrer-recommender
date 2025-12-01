
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Career Recommender", layout="centered")

st.title("🎯 AI Career Recommendation System")
st.write("Developed by **NARLA JYOTHI SWAROOPA** | NASSCOM FutureSkills AI Internship")
st.write("Rate your skill levels from 1 (low) to 10 (high).")

model = joblib.load("career_model.pkl")
encoder = joblib.load("label_encoder.pkl")

traits = [
    "Python", "Java", "C_CPP",
    "Web_Development", "Data_Analysis", "Machine_Learning",
    "Database", "Cloud_Computing", "Cybersecurity",
    "UI_UX_Design", "Business_Analysis",
    "Leadership", "Communication", "Problem_Solving",
    "Creativity", "Public_Speaking", "Teamwork",
    "Experience",
]

cols = st.columns(2)
inputs = {}
for i, t in enumerate(traits):
    with cols[i % 2]:
        inputs[t] = st.slider(t.replace("_", " "), 1.0, 10.0, 5.0)

if st.button("🎯 Recommend Career"):
    df = pd.DataFrame([[inputs[t] for t in traits]], columns=traits)

    probs = model.predict_proba(df)[0]

    class_indices = model.classes_
    career_names = encoder.inverse_transform(class_indices)

    ranked = sorted(
        zip(career_names, probs),
        key=lambda x: x[1],
        reverse=True
    )

    top3 = ranked[:3]

    best_career, best_prob = top3[0]
    st.success(f"💡 Best Match: **{best_career}** ({best_prob*100:.1f}% match)")

    st.subheader("🏆 Top 3 Career Recommendations")

    box_colors = ["#00C853", "#2962FF", "#D500F9"]
    rank_icons = ["🥇", "🥈", "🥉"]

    for idx, (career, prob) in enumerate(top3):
        color = box_colors[idx]
        icon = rank_icons[idx]

        # IMPORTANT: HTML has NO leading spaces so Streamlit doesn't treat it as code
        html = f"""
<div style="background-color:{color};
padding:18px;
border-radius:12px;
margin-bottom:15px;
color:white;">

<h3>{icon} Rank {idx+1}: {career}</h3>
<p style="font-size:17px;">Match Score: <b>{prob*100:.1f}%</b></p>

<div style="background-color:white;
height:18px;
width:100%;
border-radius:8px;
margin-top:6px;">
    <div style="background-color:black;
    height:18px;
    width:{prob*100}%;
    border-radius:8px;">
    </div>
</div>

</div>
"""
        st.markdown(html, unsafe_allow_html=True)

    st.balloons()
