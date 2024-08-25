import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Define the OutlierHandling class
class OutlierHandling(BaseEstimator, TransformerMixin):
    def __init__(self, upper_limit=0.95, lower_limit=0.05):
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def fit(self, X, y=None):
        # Calculate the limits for winsorization
        self.lower_bounds_ = np.percentile(X, self.lower_limit * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_limit * 100, axis=0)
        return self

    def transform(self, X):
        # Winsorize data
        X = np.where(X < self.lower_bounds_, self.lower_bounds_, X)
        X = np.where(X > self.upper_bounds_, self.upper_bounds_, X)
        return X

# Load the trained model
model = joblib.load('model/final_model.pkl')

# Define function for single prediction with probability
def predict_single_instance(data):
    df = pd.DataFrame([data])
    prediction_proba = model.predict_proba(df)
    churn_proba = prediction_proba[0][1]  # Probability of churning
    return churn_proba

# Define function to predict from CSV
def predict_from_csv(file):
    data = pd.read_csv(file)
    predictions = model.predict(data)
    data['Churn_Prediction'] = predictions
    return data

# Streamlit app
st.title('Churn Prediction App')

# Sidebar options
st.sidebar.header('Options')
option = st.sidebar.selectbox('Select an option:', ('Predict using CSV', 'Manual input for single instance'))

if option == 'Predict using CSV':
    st.header('Upload a CSV file')
    file = st.file_uploader('Upload CSV', type=['csv'])

    if file is not None:
        predictions_df = predict_from_csv(file)
        st.write(predictions_df)

        # Option to download the predictions
        csv = predictions_df.to_csv(index=False)
        st.download_button('Download Predictions', csv, 'predictions.csv', 'text/csv')

elif option == 'Manual input for single instance':
    st.header('Input Customer Data')

    # Input fields for manual input
    CustomerID = st.text_input('CustomerID')
    Tenure = st.number_input('Tenure', min_value=0.0, max_value=100.0)
    PreferredLoginDevice = st.selectbox('PreferredLoginDevice', ['Mobile Phone', 'Phone'])
    CityTier = st.number_input('CityTier', min_value=1, max_value=3)
    WarehouseToHome = st.number_input('WarehouseToHome', min_value=0.0, max_value=100.0)
    PreferredPaymentMode = st.selectbox('PreferredPaymentMode', ['Debit Card', 'UPI', 'CC'])
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    HourSpendOnApp = st.number_input('HourSpendOnApp', min_value=0.0, max_value=24.0)
    NumberOfDeviceRegistered = st.number_input('NumberOfDeviceRegistered', min_value=1, max_value=10)
    PreferedOrderCat = st.selectbox('PreferedOrderCat', ['Laptop & Accessory', 'Mobile'])
    SatisfactionScore = st.number_input('SatisfactionScore', min_value=0, max_value=10)
    MaritalStatus = st.selectbox('MaritalStatus', ['Single', 'Married'])
    NumberOfAddress = st.number_input('NumberOfAddress', min_value=1, max_value=10)
    Complain = st.selectbox('Complain', [0, 1])
    OrderAmountHikeFromlastYear = st.number_input('OrderAmountHikeFromlastYear', min_value=0.0, max_value=100.0)
    CouponUsed = st.number_input('CouponUsed', min_value=0.0, max_value=100.0)
    OrderCount = st.number_input('OrderCount', min_value=0.0, max_value=100.0)
    DaySinceLastOrder = st.number_input('DaySinceLastOrder', min_value=0.0, max_value=365.0)
    CashbackAmount = st.number_input('CashbackAmount', min_value=0.0, max_value=1000.0)

    input_data = {
        'CustomerID': CustomerID,
        'Tenure': Tenure,
        'PreferredLoginDevice': PreferredLoginDevice,
        'CityTier': CityTier,
        'WarehouseToHome': WarehouseToHome,
        'PreferredPaymentMode': PreferredPaymentMode,
        'Gender': Gender,
        'HourSpendOnApp': HourSpendOnApp,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
        'PreferedOrderCat': PreferedOrderCat,
        'SatisfactionScore': SatisfactionScore,
        'MaritalStatus': MaritalStatus,
        'NumberOfAddress': NumberOfAddress,
        'Complain': Complain,
        'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
        'CouponUsed': CouponUsed,
        'OrderCount': OrderCount,
        'DaySinceLastOrder': DaySinceLastOrder,
        'CashbackAmount': CashbackAmount
    }

    if st.button('Predict'):
        churn_proba = predict_single_instance(input_data)
        st.write(f'There is a {churn_proba * 100:.2f}% chance the customer will churn.')
