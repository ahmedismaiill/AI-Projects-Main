import joblib
import pandas as pd
import streamlit as st

# Load Model
model_Bagging = joblib.load('xgboost.pkl')
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Build Streamlit
st.cache_data.clear()

st.set_page_config(page_title="Loan Approval App",page_icon='🚨')

st.title('🏦 Loan Approval Prediction App')
st.markdown(
    """
    <div style="background-color:#000000; padding:10px; border-radius:5px">
        <h4 style="color:#faf7f7;">This app analyzes a dataset of loan applicants to explore approval patterns and predict whether a loan will be approved based on features like age, income, loan amount, interest rate, credit history, and more.🤖💼 
        </h4>
    </div>
    """,
    unsafe_allow_html=True
)
st.image("original-d7f03bc0db757267c351a8294f1e8092.gif",use_container_width=True)

person_home_ownership_list_Selection = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
loan_intent_list_selection = [0, 1, 2, 3, 4, 5]
previous_loan_defaults_on_file_list_selection = ['Yes','No']



with st.container(height=1050):
    col1 , col2 = st.columns(2)

    with col1:
        st.write("### Person Age")
        st.caption("Age of the person")
        age = st.slider("Select Age:",0,60,24)

        st.write("### Person Income")
        st.caption("Annual Income")
        person_income = st.slider("Select Annual Income",4000,200000,8000)       
        
        st.write("### Person Employee Experience")
        st.caption("Years of employment experience")
        person_emp_exp = st.slider("Select Person Employee Experience",0,40,3)   

        st.write("### Person Home Ownership")
        st.caption("Home ownership status (e.g., rent, own, mortgage)")
        person_home_ownership = st.selectbox("Select:",person_home_ownership_list_Selection,key='person_home_ownership')

        st.write("### Loan Amount")
        st.caption("Loan Amount Requested")
        loan_amnt = st.slider("Select Loan Amount",500,30000,15000)   

    with col2:
        st.write("### Loan Interest Rate")
        st.caption("Loan Interest Rate")
        loan_int_rate = st.slider("Select",4,20,8) 
       
        st.write("### Credit Bureau Person Credit History Length")
        st.caption("Length of Credit History in Years")
        cb_person_cred_hist_length = st.slider("Select",2,30,6) 
       
        st.write("### Loan Intent")
        st.caption("Purpose of the loan")
        loan_intent = st.selectbox("Select:",loan_intent_list_selection,key='loan_intent')

        loan_percent_income = loan_amnt / person_income
        st.metric(label="Loan Amount as a Percentage of Annual Income", value=loan_percent_income,label_visibility='visible',border=True)




person_home_ownership_encode = ['RENT','MORTGAGE','OWN','OTHER']
encode_person_home_ownership = [3,0,2,1]
convert_person_home_ownership_encode = dict(zip(person_home_ownership_encode, encode_person_home_ownership))
							
try:
 df = pd.DataFrame({
          'person_age':age,
          'person_income':person_income,
          'person_emp_exp':person_emp_exp,
          'person_home_ownership':convert_person_home_ownership_encode[person_home_ownership],
          'loan_amnt':loan_amnt,
          'loan_intent':loan_intent,
          'loan_int_rate':loan_int_rate,
          'loan_percent_income':loan_percent_income,    
          'cb_person_cred_hist_length':cb_person_cred_hist_length,
          },index=[0])
except ValueError:
    st.error("Please ensure all numerical fields are correctly filled! ❌")

print(df)


def predict_heart_failure(df):
    prediction = model_Bagging.predict(df)
    prediction_prob = model_Bagging.predict_proba(df)  
    return prediction , prediction_prob
prediction , prediction_prob = predict_heart_failure(df)
print(prediction)
print(prediction_prob)
rejected_percent = str((prediction_prob[0,0])*100)[:5] + '%'
approved_percent = str((prediction_prob[0,1])*100)[:5] + '%'
print(approved_percent)

with st.sidebar:
    st.title('🔍 Loan Approved or Not')
    st.divider()  
    b = st.button("Start", icon='🚨', use_container_width=True)
    st.divider()
    if b:
      if prediction == 1:
          st.badge(f"Approved {approved_percent}",color='green',icon='✅')
          st.image('approved.gif')
      else:
         st.badge(f"Rejected {rejected_percent}",color='red',icon='❌')
         st.image('rejected.gif')
