import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess data
df = pd.read_csv("dream_data.csv")
df.columns = df.columns.str.lower()

le_sound = LabelEncoder()
le_mood = LabelEncoder()
le_goal = LabelEncoder()
le_dream = LabelEncoder()

df['sound'] = le_sound.fit_transform(df['sound'])
df['mood'] = le_mood.fit_transform(df['mood'])
df['goal'] = le_goal.fit_transform(df['goal'])
df['dream'] = le_dream.fit_transform(df['dream'])

# Split features and target
x = df[['sound', 'mood', 'goal']]
y = df['dream']

# Train model
model = RandomForestClassifier()
model.fit(x, y)

# Save model and encoders
joblib.dump(model, 'model.pkl')
joblib.dump(le_sound, 'le_sound.pkl')
joblib.dump(le_mood, 'le_mood.pkl')
joblib.dump(le_goal, 'le_goal.pkl')
joblib.dump(le_dream, 'le_dream.pkl')

print("✅ Model and encoders saved successfully!")

