import pandas as pd
import streamlit as st

import pickle as pk


import os

model_path = 'House_prediction_model.pkl'
if os.path.exists(model_path):
    with open(model_path,'rb') as f:
        model = pk.load(f)
else:
    st.error("Model file not found! Make sure 'House_prediction_model.pkl' is in the same folder as app.py")
    st.stop()


st.header('Bangalore House Prices Predictor')
data = pd.read_csv('Bengaluru_House_Data.csv')

loc = st.selectbox('Choose the location', data['location'].dropna().astype(str).unique())
sqft = st.number_input('Enter Total sqft')
beds = st.number_input('Enter Number of Bedrooms')
bath = st.number_input('Enter Number of baths')
balc = st.number_input('Enter Number of Balconies')

if st.button('Predict Price'):
    input_df = pd.DataFrame([[loc, sqft, bath, balc, beds]],
                            columns=['location','total_sqft','bath','balcony','bedrooms'])
    output = model.predict(input_df)
    st.success('Price of House is ₹' + str(output[0]*100000))
