import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# df = pd.read_csv("cleaned_loan2.csv")
# st.title("Loan Approval Prediction App")
# st.sidebar.header("User Input Features")

def user_input_features():
    ApplicantIncome=st.sidebar.slider("Applicant Income",min_value=150,max_value=10500,value=5000)
    CoapplicantIncome = st.sidebar.slider("CoapplicantIncome", min_value=150, max_value=6000, value=500)
    LoanAmount = st.sidebar.slider("LoanAmount", min_value=0, max_value=500, value=250)
    Loan_Amount_Term = st.sidebar.slider("Loan_Amount_Term", min_value=0, max_value=500, value=250)

    data = {
        "ApplicantIncome" : ApplicantIncome,
        "CoapplicantIncome" : CoapplicantIncome,
        "LoanAmount" : LoanAmount,
        "Loan_Amount_Term" : Loan_Amount_Term
    }
    features = pd.DataFrame(data,index = [0])
    return features

df=user_input_features()

st.subheader('User Input Parameters')
st.write(df)


Loan = pd.read_csv("cleaned_loan2.csv")
X = Loan[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]]
y = Loan["Loan_Status"]

clf = RandomForestClassifier()
clf.fit(X,y)
prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(X)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_prob)


