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
import base64

# Page title
st.set_page_config(page_title='IZI MACHINE LEARNING', page_icon='🤖')
st.set_page_config(page_title='IZI MACHINE LEARNING', page_icon='🤖', layout='wide')
st.title('🤖 IZI MACHINE LEARNING')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- Drug solubility data set
  ''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- Pandas for data wrangling
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
- Altair for chart creation
- Streamlit for user interface
  ''', language='markdown')
    ''', language='markdown')


# Sidebar for accepting input parameters
@@ -41,11 +43,12 @@
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)
      

    # Download example data
    @st.cache_data
    def convert_df(input_df):
        return input_df.to_csv(index=False).encode('utf-8')

    example_csv = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    csv = convert_df(example_csv)
    st.download_button(
@@ -80,9 +83,13 @@ def convert_df(input_df):

    sleep_time = st.slider('Sleep time', 0, 3, 0)

# Model selection
st.sidebar.header('3. Select Model')
model_type = st.sidebar.selectbox("Choose a model type", ("Random Forest", "Linear Regression"))

# Initiate the model building process
if uploaded_file or example_data: 
    with st.status("Running ...", expanded=True) as status:
    with st.spinner("Running ..."):

        st.write("Loading data ...")
        time.sleep(sleep_time)
@@ -99,25 +106,29 @@ def convert_df(input_df):
        st.write("Model training ...")
        time.sleep(sleep_time)

        if parameter_max_features == 'all':
            parameter_max_features = None
            parameter_max_features_metric = X.shape[1]

        rf = RandomForestRegressor(
                n_estimators=parameter_n_estimators,
                max_features=parameter_max_features,
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf,
                random_state=parameter_random_state,
                criterion=parameter_criterion,
                bootstrap=parameter_bootstrap,
                oob_score=parameter_oob_score)
        rf.fit(X_train, y_train)
        if model_type == "Random Forest":
            if parameter_max_features == 'all':
                parameter_max_features = None
                parameter_max_features_metric = X.shape[1]

            model = RandomForestRegressor(
                    n_estimators=parameter_n_estimators,
                    max_features=parameter_max_features,
                    min_samples_split=parameter_min_samples_split,
                    min_samples_leaf=parameter_min_samples_leaf,
                    random_state=parameter_random_state,
                    criterion=parameter_criterion,
                    bootstrap=parameter_bootstrap,
                    oob_score=parameter_oob_score)
        elif model_type == "Linear Regression":
            model = LinearRegression()

        model.fit(X_train, y_train)

        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
@@ -129,20 +140,12 @@ def convert_df(input_df):
        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)
        parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])
        #if 'Mse' in parameter_criterion_string:
        #    parameter_criterion_string = parameter_criterion_string.replace('Mse', 'MSE')
        rf_results = pd.DataFrame(['Random forest', train_mse, train_r2, test_mse, test_r2]).transpose()
        rf_results.columns = ['Method', f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2']
        # Convert objects to numerics
        for col in rf_results.columns:
            rf_results[col] = pd.to_numeric(rf_results[col], errors='ignore')
        # Round to 3 digits
        rf_results = rf_results.round(3)
        results = pd.DataFrame([['Random forest' if model_type == "Random Forest" else 'Linear Regression', train_mse, train_r2, test_mse, test_r2]]).transpose()
        results.columns = ['Method', f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2']
        results = results.round(3)

    status.update(label="Status", state="complete", expanded=False)

    # Display data info
    st.header('Input data', divider='rainbow')
    st.header('Input data')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
@@ -189,33 +192,35 @@ def convert_df(input_df):
                )

    # Display model parameters
    st.header('Model parameters', divider='rainbow')
    st.header('Model parameters')
    parameters_col = st.columns(3)
    parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
    parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
    parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")
    if model_type == "Random Forest":
        parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
        parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")

    # Display feature importance plot
    importances = rf.feature_importances_
    feature_names = list(X.columns)
    forest_importances = pd.Series(importances, index=feature_names)
    df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})

    bars = alt.Chart(df_importance).mark_bar(size=40).encode(
             x='value:Q',
             y=alt.Y('feature:N', sort='-x')
           ).properties(height=250)

    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.header('Model performance', divider='rainbow')
        st.dataframe(rf_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
    with performance_col[2]:
        st.header('Feature importance', divider='rainbow')
        st.altair_chart(bars, theme='streamlit', use_container_width=True)
    # Display feature importance plot if Random Forest
    if model_type == "Random Forest":
        importances = model.feature_importances_
        feature_names = list(X.columns)
        forest_importances = pd.Series(importances, index=feature_names)
        df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})

        bars = alt.Chart(df_importance).mark_bar(size=40).encode(
                x='value:Q',
                y=alt.Y('feature:N', sort='-x')
            ).properties(height=250)

        performance_col = st.columns((2, 0.2, 3))
        with performance_col[0]:
            st.header('Model performance')
            st.dataframe(results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
        with performance_col[2]:
            st.header('Feature importance')
            st.altair_chart(bars, theme='streamlit', use_container_width=True)

    # Prediction results
    st.header('Prediction results', divider='rainbow')
    st.header('Prediction results')
    s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
    df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    df_train['class'] = 'train'
        
    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
    df_test['class'] = 'test'
    
    df_prediction = pd.concat([df_train, df_test], axis=0)
    
    prediction_col = st.columns((2, 0.2, 3))
    
    # Display dataframe
    with prediction_col[0]:
        st.dataframe(df_prediction, height=320, use_container_width=True)
    # Display scatter plot of actual vs predicted values
    with prediction_col[2]:
        scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
                        x='actual',
                        y='predicted',
                        color='class'
                  )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)


# Ask for CSV upload if none is detected
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
