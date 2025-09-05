import pandas as pd
import pickle as pk
import streamlit as st

model = pk.load(open('House_prediction_model.pkl','rb'))

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
