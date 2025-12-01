import os
import random
import textwrap
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("🚀 Starting Career Recommendation System Build...")

# === Folder Setup ===
base = os.getcwd()
print(f"📁 Using folder: {base}")

# === Define Traits (Personality + Skills + Experience) ===
columns = [
    # Big Five personality
    'O_score', 'C_score', 'E_score', 'A_score', 'N_score',

    # Technical skills
    'Python', 'Java', 'C_CPP',
    'Web_Development', 'Data_Analysis', 'Machine_Learning',
    'Database', 'Cloud_Computing', 'Cybersecurity',
    'UI_UX_Design', 'Business_Analysis',

    # Soft skills
    'Leadership', 'Communication', 'Problem_Solving',
    'Creativity', 'Public_Speaking', 'Teamwork',

    # Overall experience
    'Experience'
]

# === Generate Dataset ===
print("🧩 Generating dataset with more careers...")

careers = {
    # Tech careers
    "Software Engineer": {
        "Python": (7, 10),
        "Java": (7, 10),
        "C_CPP": (7, 10),
        "Problem_Solving": (8, 10),
        "Teamwork": (7, 10),
    },
    "Frontend Developer": {
        "Web_Development": (8, 10),
        "UI_UX_Design": (7, 10),
        "Creativity": (8, 10),
        "O_score": (7, 10),
    },
    "Backend Developer": {
        "Python": (7, 10),
        "Java": (7, 10),
        "Database": (8, 10),
        "Cloud_Computing": (7, 10),
        "Problem_Solving": (8, 10),
    },
    "Full Stack Developer": {
        "Web_Development": (8, 10),
        "Python": (7, 10),
        "Database": (7, 10),
        "Problem_Solving": (8, 10),
        "Teamwork": (7, 10),
    },
    "Data Scientist": {
        "Python": (8, 10),
        "Data_Analysis": (8, 10),
        "Machine_Learning": (8, 10),
        "Problem_Solving": (8, 10),
    },
    "Database Administrator": {
        "Database": (8, 10),
        "C_score": (8, 10),
        "Problem_Solving": (7, 10),
        "Attention_to_detail": (0, 0),  # just for concept, not a column
    },
    "Cloud Engineer": {
        "Cloud_Computing": (8, 10),
        "Python": (6, 9),
        "Problem_Solving": (7, 10),
        "C_score": (7, 10),
    },
    "Cybersecurity Analyst": {
        "Cybersecurity": (8, 10),
        "Problem_Solving": (8, 10),
        "C_CPP": (6, 9),
        "C_score": (7, 10),
    },
    "Web Developer": {
        "Web_Development": (8, 10),
        "UI_UX_Design": (7, 10),
        "Creativity": (8, 10),
        "Communication": (6, 9),
    },

    # Analysis / business careers
    "Business Analyst": {
        "Business_Analysis": (8, 10),
        "Data_Analysis": (7, 10),
        "Communication": (7, 10),
        "Public_Speaking": (6, 9),
    },
    "Project Manager": {
        "Leadership": (8, 10),
        "Communication": (8, 10),
        "Teamwork": (8, 10),
        "Experience": (7, 10),
    },

    # People & management careers
    "HR Manager": {
        "Communication": (8, 10),
        "Public_Speaking": (7, 10),
        "Leadership": (7, 10),
        "Teamwork": (8, 10),
        "A_score": (7, 10),
    },
    "Teacher": {
        "Communication": (8, 10),
        "Public_Speaking": (8, 10),
        "A_score": (7, 10),
        "Experience": (6, 10),
    },

    # Creative careers
    "UI/UX Designer": {
        "UI_UX_Design": (8, 10),
        "Creativity": (8, 10),
        "O_score": (8, 10),
        "Web_Development": (6, 9),
    },
    "Graphic Designer": {
        "Creativity": (8, 10),
        "UI_UX_Design": (7, 10),
        "O_score": (8, 10),
        "E_score": (6, 9),
    },
    "Artist": {
        "O_score": (8, 10),
        "Creativity": (9, 10),
        "E_score": (7, 10),
    },
    "Journalist": {
        "Communication": (8, 10),
        "Public_Speaking": (7, 10),
        "O_score": (7, 10),
        "Creativity": (7, 10),
    },

    # Classic careers from earlier
    "Doctor": {
        "C_score": (8, 10),
        "A_score": (8, 10),
        "Communication": (7, 10),
        "Experience": (7, 10),
    },
    "Lawyer": {
        "Communication": (8, 10),
        "Public_Speaking": (8, 10),
        "E_score": (7, 10),
        "A_score": (7, 10),
    },
    "Mechanical Engineer": {
        "C_CPP": (7, 10),
        "Problem_Solving": (8, 10),
        "Data_Analysis": (6, 9),
    },
    "IAS Officer": {
        "Leadership": (8, 10),
        "Communication": (8, 10),
        "C_score": (7, 10),
        "A_score": (7, 10),
        "Public_Speaking": (7, 10),
    },
}

rows = []
for career, prefs in careers.items():
    for _ in range(60):  # number of samples per career
        row = {}
        for col in columns:
            if col in prefs:
                low, high = prefs[col]
                row[col] = round(random.uniform(low, high), 1)
            else:
                # general mid-range values for non-key traits
                row[col] = round(random.uniform(3, 8), 1)
        row["Career"] = career
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("Data_final.csv", index=False)
print("✅ Dataset created: Data_final.csv")

# === Train Model ===
print("🤖 Training model...")
X = df.drop("Career", axis=1)
y = df["Career"]

enc = LabelEncoder()
y_enc = enc.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    random_state=42
)
model.fit(X, y_enc)

joblib.dump(model, "career_model.pkl")
joblib.dump(enc, "label_encoder.pkl")
print("✅ Model trained and saved!")

# === Streamlit App ===
print("💻 Creating app.py...")

app_code = textwrap.dedent('''
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
''')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("✅ app.py created!")

# === Requirements ===
open("requirements.txt", "w").write("pandas\nscikit-learn\nstreamlit\njoblib\n")
print("✅ requirements.txt created!")

# === README ===
readme = """Career Recommendation System
Developed by: NARLA JYOTHI SWAROOPA

Steps to Run:
1. pip install -r requirements.txt
2. python -m streamlit run app.py

Description:
Predicts the most suitable career based on personality traits, technical skills, soft skills, and experience.
Tools: Python, scikit-learn, Streamlit.
"""
open("README.txt", "w").write(readme)
print("✅ README created!")

print("\\n🎉 Project built successfully!")
print("Now run these commands next:")
print("pip install -r requirements.txt")
print("python -m streamlit run app.py")
