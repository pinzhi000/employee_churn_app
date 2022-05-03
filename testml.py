import streamlit as st
import pandas as pd
import numpy as np 
import sklearn
import os 

# pickle! 
import pickle

# create two pages in streamlit app 
    # page 1: homepage

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Predict_Churn'])

if app_mode == 'Home':
    st.title("Employee Prediction")
    st.markdown("Dataset :")
    
    # set path 
    path = os.path.dirname(__file__)

    df = pd.read_csv(path + '/emp_analytics.csv')
    st.write(df.head())

    # page 2: prediction page -- interaction with algo 
elif app_mode == 'Predict_Churn':

    st.subheader("Fill in employee details to get prediction")
    st.sidebar.header("Other details :")

    # end user selects inputs
    satisfaction_level = st.number_input("satisfaction level", min_value = 0.0, max_value = 1.0)
    average_monthly_hours = st.number_input("average monthly hours")
    promotion_last_5years = st.number_input("number of promotions in last 5 years")

    # salary binary selection w/radio buttons   
    prop = {'salary_low': 1, 'salary_medium': 2, 'salary_high': 3}
    salary = st.sidebar.radio("Select Salary ", tuple(prop.keys()))

    salary_low = 0
    salary_medium = 0
    salary_high = 0

    if salary == 'High':
        salary_high = 1
    elif salary == 'Low':
        salary_low = 1
    else:
        salary_medium = 1 


    # store end user webpage selections as dictionary --> store in results variable 
        # create variables as dictionary and assign to features variable 
    subdata = {'satisfaction_level': satisfaction_level, 
               'average_monthly_hours': average_monthly_hours,
               'promotion_last_5years': promotion_last_5years,
               'salary': [salary_low, salary_medium, salary_high]}  

    features = [satisfaction_level, average_monthly_hours, promotion_last_5years, subdata['salary'][0], subdata['salary'][1], subdata['salary'][2]]
    
    # "results" will be fed into pickle model 
    results = np.array(features).reshape(1, -1)

    if st.button("Predict"):

        picklefile = open("emp-model.pkl", "rb")
        model = pickle.load(picklefile)

        # make prediction 
        prediction = model.predict(results)

        if prediction[0] == 0:
            st.success('Employee will not churn')
        elif prediction[0] == 1:
            st.error('Employee will churn')








    






