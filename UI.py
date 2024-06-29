import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


model = pickle.load(open('model.pkl', 'rb'))


scaler_mean = np.array([287.63598, 5.48216, 153.93083, 2.05321])
scaler_scale = np.array([1041.25522, 3.15713, 685.41759, 4.11907])

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale


cluster_names = {
    0: "High Value, Frequent Buyers",
    1: "Low Value, Occasional Buyers",
    2: "High Quantity, Low Value Buyers",
    3: "Average Buyers"
}


st.title("Customer Segmentation")


total_spent = st.number_input("Total Spent", min_value=0.0, format="%.2f")
num_invoices = st.number_input("Number of Invoices", min_value=1, format="%d")
quantity = st.number_input("Total Quantity", min_value=0, format="%d")
avg_unit_price = st.number_input("Average Unit Price", min_value=0.0, format="%.2f")


if st.button("Predict Cluster"):
    
    input_data = np.array([[total_spent, num_invoices, quantity, avg_unit_price]])
    
    
    input_data_scaled = scaler.transform(input_data)
    
    
    cluster = model.predict(input_data_scaled)[0]
    
    
    cluster_name = cluster_names.get(cluster, "Unknown Cluster")
    
    
    st.write(f"The customer belongs to the cluster: **{cluster_name}**")
