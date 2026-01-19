import pickle as pk
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('./reg_model.pkl', 'rb') as f:
  model = pk.load(f)


with open('scaler.pkl', 'rb') as f:
    scaler = pk.load(f)


st.title("💹 Student Performance Predictor App💹")
st.write("Predict your score based on the following questions ❓")

st.info("Fill in the form below")

with st.form(key = 'score'):
    # ['Hours Studied', 'Previous Scores', 
    # 'Extracurricular Activities',
    #    'Sleep Hours', 'Sample Question Papers Practiced']
    hours_study = st.number_input("How many Hours do you study per day ?", min_value=1, max_value=10)
    prev_score = st.number_input("What is your previous score?", min_value = 0, max_value = 100)
    ext_act = st.selectbox("Do you perform extra curriculum activities?", options=["Yes", "No"])
    sleep = st.number_input("How many hours do you sleep?", min_value = 1, max_value = 10)
    sample_que = st.number_input("Number of sample questions you practice?", min_value = 1, max_value=10)
    btn = st.form_submit_button("Predict")
    

    if btn:
        data = {'Hours Studied':[hours_study], 'Previous Scores':[prev_score], 
                'Extracurricular Activities': [ext_act], 'Sleep Hours': [sleep], 
                'Sample Question Papers Practiced': [sample_que]}
        df = pd.DataFrame(data)
        df['Extracurricular Activities'] = df['Extracurricular Activities'].map({"Yes": 1, "No":0})
        df = scaler.transform(df)
        
        st.success(f"Based on the info you entered, your future score would be around {model.predict(df)} ")
