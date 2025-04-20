import streamlit as st
import joblib

# Load saved model and encoders
model = joblib.load('model.pkl')
le_sound = joblib.load('le_sound.pkl')
le_mood = joblib.load('le_mood.pkl')
le_goal = joblib.load('le_goal.pkl')
le_dream = joblib.load('le_dream.pkl')  # For decoding prediction

st.title("💤 Sleep Sound Classifier & Dream Suggestor")

# User Inputs
sound = st.selectbox("Select Sound Type 🎵", le_sound.classes_)
mood = st.selectbox("Select Your Mood 😊", le_mood.classes_)
goal = st.selectbox("Select Sleep Goal 🛌", le_goal.classes_)

if st.button("Predict Dream Type 🔮"):
    # Encode input
    input_data = [
        le_sound.transform([sound])[0],
        le_mood.transform([mood])[0],
        le_goal.transform([goal])[0]
    ]
    prediction = model.predict([input_data])[0]
    dream_type = le_dream.inverse_transform([prediction])[0]

    st.success(f"🌙 You might experience a **'{dream_type}'** dream tonight!")
