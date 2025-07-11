import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("🌧️ Rainfall Forecasting - Mumbai")
st.subheader("Compare Linear Regression vs KNN Regression Predictions")

# Load saved models and scaler
with open('rainfall_forecasting_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Input Features
st.sidebar.header("Input Monthly Rainfall Data (mm)")

months = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June',
          'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

rainfall_inputs = []
for month in months:
    val = st.sidebar.number_input(f"{month} Rainfall (mm)", min_value=0.0, value=100.0)
    rainfall_inputs.append(val)

# Calculate seasonal totals
Rainy_Season_Total = rainfall_inputs[5] + rainfall_inputs[6] + rainfall_inputs[7]  # June + July + Aug
Pre_Monsoon_Total = rainfall_inputs[2] + rainfall_inputs[3] + rainfall_inputs[4]   # Mar + April + May
Post_Monsoon_Total = rainfall_inputs[8] + rainfall_inputs[9] + rainfall_inputs[10] # Sept + Oct + Nov
Summer_Total = rainfall_inputs[2] + rainfall_inputs[3] + rainfall_inputs[4]         # Same as Pre-Monsoon

Monthly_Mean = np.mean(rainfall_inputs)
Monthly_Std = np.std(rainfall_inputs)

# Final feature array
final_features = rainfall_inputs + [Rainy_Season_Total, Pre_Monsoon_Total, Post_Monsoon_Total, Summer_Total, Monthly_Mean, Monthly_Std]
final_array = np.array(final_features).reshape(1, -1)

# Scale the inputs
scaled_features = scaler.transform(final_array)

# Prediction
if st.button('Predict Rainfall'):
    linear_pred = linear_model.predict(scaled_features)[0]
    knn_pred = knn_model.predict(scaled_features)[0]

    st.success(f"Linear Regression Predicted Rainfall: {linear_pred:.2f} mm")
    st.success(f"KNN Regression Predicted Rainfall: {knn_pred:.2f} mm")

    # Comparison Bar Chart
    st.subheader("📊 Prediction Comparison")
    models = ['Linear Regression', 'KNN Regression']
    predictions = [linear_pred, knn_pred]

    fig, ax = plt.subplots()
    bars = ax.bar(models, predictions, color=['orange', 'seagreen'], edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.2f}', ha='center', va='bottom')

    ax.set_ylabel('Predicted Total Rainfall (mm)')
    st.pyplot(fig)
