import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Load the GPT-Neo model
@st.cache_resource
def load_model():
    try:
        model = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Error loading model")
        return None

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
    - Altair for data visualization
    - Transformers for text generation
    ''', language='markdown')

# Main ML app functionality
st.header("Machine Learning Model Builder")

# Sidebar for dataset selection
with st.sidebar:
    st.header("Step 1: Select Dataset")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset preview:")
        st.write(data.head())

# Step 2: Data Preprocessing
if uploaded_file:
    st.header("Step 2: Data Preprocessing")
    st.markdown("Select features and target variable for the ML model.")
    
    all_columns = data.columns.tolist()
    features = st.multiselect("Select feature columns", all_columns)
    target = st.selectbox("Select target column", all_columns)
    
    if features and target:
        X = data[features]
        y = data[target]
        
        # Check for non-numeric data in target
        try:
            y = pd.to_numeric(y)
        except ValueError:
            st.error("The target column contains non-numeric data, which cannot be processed.")
            st.stop()

        st.write("Selected features preview:")
        st.write(X.head())
        
        st.write("Selected target preview:")
        st.write(y.head())
        
        # Split the data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            st.error("Error splitting data")
            X_train, X_test, y_train, y_test = None, None, None, None

        # Step 3: Model Building
        st.header("Step 3: Model Building")
        model_type = st.selectbox("Select model type", ["Random Forest", "Linear Regression"])

        try:
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of trees in forest", 1, 100)
                model = RandomForestRegressor(n_estimators=n_estimators)
            else:
                model = LinearRegression()

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception as e:
            logger.error(f"Error training model: {e}")
            st.error("Error training model")
            y_pred = None

        # Display results
        st.header("Step 4: Model Results")
        if y_pred is not None:
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
            st.write("R2 Score:", r2_score(y_test, y_pred))

            # Plot results
            st.subheader("Prediction vs Actual")
            chart_data = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            chart = alt.Chart(chart_data.reset_index()).mark_circle(size=60).encode(
                x='Actual',
                y='Predicted',
                tooltip=['Actual', 'Predicted']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

# Add the text generation part to the main functionality
st.header("Text Generation with GPT-Neo")
user_input = st.text_input("Enter a prompt for text generation:")
if user_input:
    model = load_model()
    if model is not None:
        try:
            response = model(user_input, max_length=50, do_sample=True)[0]['generated_text']
            st.write(response)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            st.error("Error generating text")
