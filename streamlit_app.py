import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
import openai
from transformers import pipeline

# Set your OpenAI API key here
openai.api_key = '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'
# Load the GPT model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# Page title
st.set_page_config(page_title='IZI MACHINE LEARNING', page_icon='🤖', layout='wide')
@@ -91,21 +93,15 @@ def convert_df(input_df):
st.sidebar.header('3. Select Model')
model_type = st.sidebar.selectbox("Choose a model type", ("Random Forest", "Linear Regression"), help="Select the type of machine learning model to use.")

# Chat with ChatGPT
st.sidebar.header('4. Chat with ChatGPT')
# Chat with GPT-J/GPT-Neo
st.sidebar.header('4. Chat with GPT-J/GPT-Neo')
st.sidebar.markdown('Ask any questions you have about machine learning, data science, or using this app.')
user_question = st.sidebar.text_input("Enter your question here:")
if user_question:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_question},
        ],
        max_tokens=100
    )
    st.sidebar.write("ChatGPT Response:")
    st.sidebar.write(response.choices[0].message['content'].strip())
    model = load_model()
    response = model(user_question, max_length=100, num_return_sequences=1)[0]['generated_text']
    st.sidebar.write("GPT-J/GPT-Neo Response:")
    st.sidebar.write(response.strip())

# Initiate the model building process
if 'df' in st.session_state: 
