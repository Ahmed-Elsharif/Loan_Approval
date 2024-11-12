import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
import category_encoders
import pickle


try:
    with open("Model.pkl", "rb") as f:
        global model
        model = pickle.load(f)
except ModuleNotFoundError as e:
    print(f"Module not found: {e}")


Inputs = joblib.load("Inputs.pkl")

def Make_Prediciton(Gender, Married, Dependents, Education, Self_Employed,
       ApplicantIncome, CoapplicantIncome, LoanAmount,
       Loan_Amount_Term, Credit_History, Property_Area):
    pr_df = pd.DataFrame(columns = Inputs)
    pr_df.at[0 ,'Gender' ] = Gender 
    pr_df.at[0 , 'Married'] = Married
    pr_df.at[0 , 'Dependents'] = Dependents
    pr_df.at[0 ,'Education' ] = Education
    pr_df.at[0 , 'Self_Employed'] =Self_Employed
    pr_df.at[0 , 'ApplicantIncome' ] =ApplicantIncome
    pr_df.at[0 , 'CoapplicantIncome'] = CoapplicantIncome
    pr_df.at[0 , 'LoanAmount'] = LoanAmount
    pr_df.at[0 , 'Loan_Amount_Term'] = Loan_Amount_Term
    pr_df.at[0 , 'Credit_History'] = Credit_History
    pr_df.at[0 , 'Property_Area'] = Property_Area

    prediction = model.predict(pr_df)
    return prediction[0]

def main():
    st.title("Loan_Approval Prediction")
    Gender = st.selectbox("Gender" , ['Male', 'Female'])
    Married = st.selectbox("Married" , ['Yes', 'No'])
    Dependents = st.selectbox("Dependents" ,[0 , 1 , 2 , 3] )
    Education = st.selectbox("Education" ,['Graduate', 'Not Graduate'] )
    Self_Employed = st.selectbox("Self_Employed" , ['No', 'Yes'])
    ApplicantIncome = st.slider("ApplicantIncome" ,  min_value=0, max_value=85, value=0, step=1)
    CoapplicantIncome = st.slider("CoapplicantIncome" ,  min_value=0, max_value=35, value=0, step=1)
    LoanAmount = st.slider("LoanAmount" ,  min_value=9, max_value=600, value=9, step=10)
    Loan_Amount_Term = st.slider("Loan_Amount_Term" ,  min_value=36, max_value=480, value=36, step=3)
    Credit_History = st.selectbox("Credit_History" ,[ 1., 0.] )
    Property_Area = st.selectbox("Property_Area" ,['Rural', 'Urban', 'Semiurban'] )

    if st.button("Predict")  :
        result = Make_Prediciton(Gender, Married, Dependents, Education, Self_Employed,
       ApplicantIncome, CoapplicantIncome, LoanAmount,
       Loan_Amount_Term, Credit_History, Property_Area)
        list_success = [ "N", "Y"]
        return st.text(f"Your Restaurant will {list_success[result]}")
main()
