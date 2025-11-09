import os, random, textwrap, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("🚀 Starting Career Recommendation System Build...")

# === Folder Setup ===
base = os.getcwd()
print(f"📁 Using folder: {base}")

# === Generate Dataset ===
print("🧩 Generating dataset...")
careers = {
    "Software Engineer": {"Numerical Aptitude": (8,10), "Abstract Reasoning": (8,10)},
    "Data Scientist": {"Numerical Aptitude": (8,10), "Abstract Reasoning": (7,10)},
    "Teacher": {"A_score": (7,10), "Verbal Reasoning": (8,10)},
    "Doctor": {"Perceptual Aptitude": (8,10), "Verbal Reasoning": (7,10)},
    "Lawyer": {"Verbal Reasoning": (8,10), "A_score": (7,9)},
    "Mechanical Engineer": {"Spatial Aptitude": (8,10), "Numerical Aptitude": (7,10)},
    "Artist": {"O_score": (8,10), "E_score": (7,10)},
    "IAS Officer": {"Verbal Reasoning": (8,10), "C_score": (7,10), "A_score": (7,10)},
}

rows = []
for career, prefs in careers.items():
    for _ in range(60):  # ~480 rows total
        row = {}
        for col in ['O_score','C_score','E_score','A_score','N_score',
                    'Numerical Aptitude','Spatial Aptitude','Perceptual Aptitude',
                    'Abstract Reasoning','Verbal Reasoning']:
            if col in prefs:
                low, high = prefs[col]
                row[col] = round(random.uniform(low, high), 1)
            else:
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
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X, y_enc)
joblib.dump(model, "career_model.pkl")
joblib.dump(enc, "label_encoder.pkl")
print("✅ Model trained and saved!")

# === Streamlit App ===
print("💻 Creating app.py...")
app_code = textwrap.dedent('''
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
Predicts the most suitable career based on aptitude and personality traits.
Tools: Python, scikit-learn, Streamlit.
"""
open("README.txt", "w").write(readme)
print("✅ README created!")

print("\n🎉 Project built successfully!")
print("Now run these commands next:")
print("pip install -r requirements.txt")
print("python -m streamlit run app.py")
