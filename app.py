import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Bank Customer Prediction AI", layout="wide")

# 1. Loading the Saved Models
@st.cache_resource
def load_models():
    # Make sure these files are in the same folder
    rf_model = joblib.load('bank_conversion_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans.pkl')
    return rf_model, model_columns, scaler, kmeans

# Loading models AFTER setting page config
rf_model, model_columns, scaler, kmeans = load_models()

# 2. App Title & Introduction
st.title("🏦 AI-Powered Bank Customer Segmentation & Prediction")
st.markdown("""
    **Welcome!** This tool helps bank managers decide:
    1.  **Prediction:** Will a customer subscribe to a Term Deposit? 
    2.  **Segmentation:** Which customer group do they belong to? (Targeting Strategy)
""")

# 3. Sidebar Inputs
st.sidebar.header("📝 Enter Customer Details")

# -- Numeric Inputs --
age = st.sidebar.slider("Age", 18, 95, 30)
balance = st.sidebar.number_input("Avg. Yearly Balance (€)", min_value=-5000, max_value=100000, value=1500)
day_of_week = st.sidebar.slider("Day of Month (1-31)", 1, 31, 15)
campaign = st.sidebar.slider("Number of Contacts (Campaign)", 1, 20, 1)
pdays = st.sidebar.number_input("Days since last contact (-1 if new)", value=-1)
previous = st.sidebar.number_input("Previous Contacts", value=0)

# -- Categorical Inputs --
job = st.sidebar.selectbox("Job Type", 
    ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 
     'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])

marital = st.sidebar.selectbox("Marital Status", ['married', 'divorced', 'single'])
education = st.sidebar.selectbox("Education Level", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.sidebar.selectbox("Has Credit in Default?", ['no', 'yes'])
housing = st.sidebar.selectbox("Has Housing Loan?", ['no', 'yes'])
loan = st.sidebar.selectbox("Has Personal Loan?", ['no', 'yes'])
contact = st.sidebar.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
month = st.sidebar.selectbox("Last Contact Month", 
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
poutcome = st.sidebar.selectbox("Previous Campaign Outcome", ['unknown', 'failure', 'other', 'success'])

# 4. Processing Logic
def preprocess_input():
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    input_df['age'] = age
    input_df['balance'] = balance
    input_df['day_of_week'] = day_of_week
    input_df['campaign'] = campaign
    input_df['pdays'] = pdays
    input_df['previous'] = previous
    
    cat_inputs = {
        'job': job, 'marital': marital, 'education': education, 
        'default': default, 'housing': housing, 'loan': loan,
        'contact': contact, 'month': month, 'poutcome': poutcome
    }
    
    for col, val in cat_inputs.items():
        dummy_name = f"{col}_{val}"
        if dummy_name in input_df.columns:
            input_df[dummy_name] = 1
            
    return input_df

# 5. Prediction & Results
if st.button("🚀 Analyze Customer"):
    user_data = preprocess_input()
    
    # -- Model 1: Prediction --
    pred_prob = rf_model.predict_proba(user_data)[0][1]
    
    # -- Model 2: Segmentation --
    segment_data = scaler.transform(user_data[['age', 'balance', 'campaign']])
    cluster_id = kmeans.predict(segment_data)[0]
    
    # -- Displaying Results --
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Prediction Result")
        if pred_prob > 0.5:
            st.success(f"**Likely to Subscribe! (YES)**")
            st.balloons()
        else:
            st.error(f"**Unlikely to Subscribe (NO)**")
        st.metric(label="Conversion Probability", value=f"{pred_prob:.1%}")
        
    with col2:
        st.subheader("🧩 Customer Segment")
        st.info(f"Customer belongs to **Cluster {cluster_id}**")
        
        if cluster_id == 0:
            st.write("👉 **Interpretation:** Likely a standard customer.")
        elif cluster_id == 1:
            st.write("👉 **Interpretation:** High Balance / Older Demographic (Good Target).")
        elif cluster_id == 2:
            st.write("👉 **Interpretation:** Low Balance / Younger Demographic.")
        else:
            st.write("👉 **Interpretation:** Targeted frequently in past campaigns.")

    st.markdown("---")
    st.caption("Developed by [Your Name]")
