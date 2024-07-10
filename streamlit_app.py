import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load the GPT-J model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Page title
st.set_page_config(page_title='IZI MACHINE LEARNING', page_icon='🤖', layout='wide')
st.title('🤖 IZI MACHINE LEARNING')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to build a machine learning (ML) model in an end-to-end workflow. This includes data upload, data pre-processing, ML model building, and post-model analysis.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and: 1. Select a data set, 2. Adjust the model parameters using the various slider widgets. This will initiate the ML model building process, display the model results, and allow users to download the generated models and accompanying data.')

    st.markdown('**Under the hood**')
    st.markdown('Data sets:')
    st.code('''- Drug solubility data set
    ''', language='markdown')

    st.markdown('Libraries used:')
    st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Transformers for NLP
- Streamlit for the web interface
    ''', language='markdown')

# Chat with the model
st.header("Chat with GPT-J")
user_input = st.text_input("Ask a question:")
if user_input:
    model = load_model()
    response = model(user_input, max_length=50, do_sample=True)[0]['generated_text']
    st.write(response)
