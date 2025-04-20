import streamlit as st
import joblib

# Load model and encoders
model = joblib.load('model.pkl')
le_sound = joblib.load('le_sound.pkl')
le_mood = joblib.load('le_mood.pkl')
le_goal = joblib.load('le_goal.pkl')
le_dream = joblib.load('le_dream.pkl')

st.title("🌙 Sleep Dream Predictor App")

sound = st.selectbox("Select sound:", le_sound.classes_)
mood = st.selectbox("Select mood:", le_mood.classes_)
goal = st.selectbox("Select goal:", le_goal.classes_)

if st.button("Predict Dream"):
    input_data = [[
        le_sound.transform([sound])[0],
        le_mood.transform([mood])[0],
        le_goal.transform([goal])[0]
    ]]
    
    prediction = model.predict(input_data)
    predicted_dream = le_dream.inverse_transform(prediction)[0]
    
    st.success(f"🌟 Your predicted dream: **{predicted_dream}**")

