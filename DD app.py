# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 23:53:34 2024

@author: HP
"""
import numpy as np 
import pickle 
import streamlit as st

#Loading the saved model
loaded_model = pickle.load(open('C:/Users/HP/Downloads/Deployment DD/trained_model.sav', 'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return'The person is not diabetic'
    else:
      return'The person is diabetic'



def main():
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting input data

    
    Pregnancies = st.text_input('Total Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction Value')
    Age = st.text_input('Age')
    
    
    
    
    
    # code for Prediction
    diagnosis = ''
    
    #creating a button for Prediction
    if st.button('Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagnosis)
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    

    
    
    

        