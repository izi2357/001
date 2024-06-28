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
- Altair for chart creation
- Streamlit for user interface
    ''', language='markdown')


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    st.markdown('**1. Use custom data**')
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
        label="Download example CSV",
        data=csv,
        file_name='delaney_solubility_with_descriptors.csv',
        mime='text/csv',
    )

    # Select example data
    st.markdown('**1.2. Use example data**')
    example_data = st.toggle('Load example data')
    if example_data:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.subheader('2.1. Learning Parameters')
    with st.expander('See parameters'):
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.subheader('2.2. General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.slider('Sleep time', 0, 3, 0)

# Model selection
st.sidebar.header('3. Select Model')
model_type = st.sidebar.selectbox("Choose a model type", ("Random Forest", "Linear Regression"))

# Initiate the model building process
if uploaded_file or example_data: 
    with st.spinner("Running ..."):
    
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)

        # Data validation and preprocessing
        if 'df' in locals():
            df = df.dropna()  # Drop missing values
            if df.select_dtypes(include=[np.number]).shape[1] == 0:
                st.error("Uploaded CSV does not contain numeric data.")
            else:
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_columns]

                if len(df.columns) < 2:
                    st.error("Uploaded CSV must contain at least one feature column and one target column.")
                else:
                    X = df.iloc[:,:-1]
                    y = df.iloc[:,-1]

                    st.write("Splitting data ...")
                    time.sleep(sleep_time)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)
                
                    st.write("Model training ...")
                    time.sleep(sleep_time)

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
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                        
                    st.write("Evaluating performance metrics ...")
                    time.sleep(sleep_time)
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    st.write("Displaying performance metrics ...")
                    time.sleep(sleep_time)
                    parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])
                    results = pd.DataFrame([['Random forest' if model_type == "Random Forest" else 'Linear Regression', train_mse, train_r2, test_mse, test_r2]]).transpose()
                    results.columns = ['Method', f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2']
                    results = results.round(3)
                    
                # Display data info
                st.header('Input data')
                col = st.columns(4)
                col[0].metric(label="No. of samples", value=X.shape[0], delta="")
                col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
                col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
                col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
                
                with st.expander('Initial dataset', expanded=True):
                    st.dataframe(df, height=210, use_container_width=True)
                with st.expander('Train split', expanded=False):
                    train_col = st.columns((3,1))
                    with train_col[0]:
                        st.markdown('**X**')
                        st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
                    with train_col[1]:
                        st.markdown('**y**')
                        st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
                with st.expander('Test split', expanded=False):
                    test_col = st.columns((3,1))
                    with test_col[0]:
                        st.markdown('**X**')
                        st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
                    with test_col[1]:
                        st.markdown('**y**')
                        st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

                # Zip dataset files
                df.to_csv('dataset.csv', index=False)
                X_train.to_csv('X_train.csv', index=False)
                y_train.to_csv('y_train.csv', index=False)
                X_test.to_csv('X_test.csv', index=False)
                y_test.to_csv('y_test.csv', index=False)
                
                list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
                with zipfile.ZipFile('dataset.zip', 'w') as zipF:
                    for file in list_files:
                        zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

                with open('dataset.zip', 'rb') as datazip:
                    btn = st.download_button(
                            label='Download ZIP',
                            data=datazip,
                            file_name="dataset.zip",
                            mime="application/octet-stream"
                            )
                
                # Display model parameters
                st.header('Model parameters')
                parameters_col = st.columns(3)
                parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
                if model_type == "Random Forest":
                    parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
                    parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")
                
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
        else:
            st.error("No data available for processing. Please upload a CSV file or use the example data.")

# Ask for CSV upload if none is detected
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
