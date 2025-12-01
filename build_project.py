import random
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("\n🚀 Building Career Recommendation System...\n")

# ============ FEATURES (SKILLS) ============
FEATURES = [
    "Python", "Java", "C_CPP",
    "Web_Development", "Data_Analysis", "Machine_Learning",
    "Database", "Cloud_Computing", "Cybersecurity",
    "UI_UX_Design", "Business_Analysis",
    "Leadership", "Communication", "Problem_Solving",
    "Creativity", "Public_Speaking", "Teamwork",
    "Experience",
]

# ============ CAREER PROFILES ============
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

# ============ 1. BUILD DATASET ============
rows = []
for career, prefs in CAREERS.items():
    for _ in range(30):      # 30 rows per career (fast but enough)
        row = {}
        for feat in FEATURES:
            if feat in prefs:
                low, high = prefs[feat]
                row[feat] = round(random.uniform(low, high), 1)
            else:
                row[feat] = round(random.uniform(1, 4), 1)  # low for non-key
        row["Career"] = career
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("Data_final.csv", index=False)
print("✅ Dataset created: Data_final.csv")

# ============ 2. TRAIN MODEL ============
X = df[FEATURES]
y = df["Career"]

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42
)
model.fit(X, y_enc)

joblib.dump(model, "career_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("✅ Model saved: career_model.pkl, label_encoder.pkl")

print("\n🎉 Build complete! Now you can use app.py with these files.\n")
