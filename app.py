import random
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Career Recommender", layout="centered")

st.title("🎯 AI Career Recommendation System")
st.write("Developed by **NARLA JYOTHI SWAROOPA** | NASSCOM FutureSkills AI Internship")
st.write("Rate your skill levels from 1 (low) to 10 (high).")

# ------------------------------------------------------------------
# 1) DEFINE FEATURES & CAREER PROFILES (same as before)
# ------------------------------------------------------------------
FEATURES = [
    "Python", "Java", "C_CPP",
    "Web_Development", "Data_Analysis", "Machine_Learning",
    "Database", "Cloud_Computing", "Cybersecurity",
    "UI_UX_Design", "Business_Analysis",
    "Leadership", "Communication", "Problem_Solving",
    "Creativity", "Public_Speaking", "Teamwork",
    "Experience",
]

CAREERS = {
    "Software Engineer": {
        "Python": (8,10), "Java": (7,10), "C_CPP": (7,10),
        "Problem_Solving": (8,10)
    },
    "Frontend Developer": {
        "Web_Development": (8,10), "UI_UX_Design": (8,10),
        "Creativity": (8,10)
    },
    "Backend Developer": {
        "Python": (7,10), "Java": (7,10), "Database": (8,10)
    },
    "Full Stack Developer": {
        "Web_Development": (8,10), "Python": (7,10),
        "Database": (7,10)
    },
    "Data Scientist": {
        "Python": (8,10), "Data_Analysis": (8,10),
        "Machine_Learning": (8,10)
    },
    "Cloud Engineer": {
        "Cloud_Computing": (8,10), "Python": (6,9)
    },
    "Cybersecurity Analyst": {
        "Cybersecurity": (8,10), "Problem_Solving": (8,10)
    },
    "Business Analyst": {
        "Business_Analysis": (8,10), "Data_Analysis": (7,10),
        "Communication": (7,10)
    },
    "Project Manager": {
        "Leadership": (8,10), "Teamwork": (8,10),
        "Communication": (8,10)
    },
    "Teacher": {
        "Communication": (8,10), "Public_Speaking": (8,10)
    },
    "UI/UX Designer": {
        "UI_UX_Design": (8,10), "Creativity": (9,10),
        "Web_Development": (5,7)
    },
    "Graphic Designer": {
        "Creativity": (9,10), "UI_UX_Design": (7,10)
    },
    "Journalist": {
        "Communication": (8,10), "Public_Speaking": (8,10),
        "Creativity": (7,10)
    },
    "Mechanical Engineer": {
        "Problem_Solving": (8,10), "C_CPP": (7,10)
    },
    "Doctor": {
        "Communication": (9,10), "Experience": (9,10)
    }
}

# ------------------------------------------------------------------
# 2) TRAIN MODEL INSIDE APP (CACHED → runs once per session)
# ------------------------------------------------------------------
@st.cache_resource
def train_model():
    rows = []
    for career, prefs in CAREERS.items():
        for _ in range(30):  # 30 samples per career
            row = {}
            for feat in FEATURES:
                if feat in prefs:
                    low, high = prefs[feat]
                    row[feat] = round(random.uniform(low, high), 1)
                else:
                    row[feat] = round(random.uniform(1, 4), 1)
            row["Career"] = career
            rows.append(row)

    df = pd.DataFrame(rows)

    X = df[FEATURES]
    y = df["Career"]

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42
    )
    model.fit(X, y_enc)

    return model, enc

model, encoder = train_model()

# ------------------------------------------------------------------
# 3) USER INPUT
# ------------------------------------------------------------------
cols = st.columns(2)
inputs = {}
for i, t in enumerate(FEATURES):
    with cols[i % 2]:
        inputs[t] = st.slider(t.replace("_", " "), 1.0, 10.0, 5.0)

# ------------------------------------------------------------------
# 4) PREDICTION & TOP 3 DISPLAY
# ------------------------------------------------------------------
if st.button("🎯 Recommend Career"):
    df_input = pd.DataFrame([[inputs[t] for t in FEATURES]], columns=FEATURES)

    probs = model.predict_proba(df_input)[0]

    class_indices = model.classes_
    career_names = encoder.inverse_transform(class_indices)

    ranked = sorted(zip(career_names, probs), key=lambda x: x[1], reverse=True)
    top3 = ranked[:3]

    best_career, best_prob = top3[0]
    st.success(f"💡 Best Match: **{best_career}** ({best_prob*100:.1f}% match)")

    st.subheader("🏆 Top 3 Career Recommendations")

    box_colors = ["#00C853", "#2962FF", "#D500F9"]
    rank_icons = ["🥇", "🥈", "🥉"]

    for idx, (career, prob) in enumerate(top3):
        color = box_colors[idx]
        icon = rank_icons[idx]

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
