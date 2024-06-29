import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st

# Load the data
df = pd.read_excel("./Online Retail.xlsx", engine='openpyxl')

# Data cleaning: Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Feature engineering: Create TotalSpent column
df['TotalSpent'] = df['Quantity'] * df['UnitPrice']

# Aggregate data per customer
customer_data = df.groupby('CustomerID').agg({
    'TotalSpent': 'sum',
    'InvoiceNo': 'nunique',
    'Quantity': 'sum',
    'UnitPrice': 'mean'
}).rename(columns={'InvoiceNo': 'NumInvoices', 'UnitPrice': 'AvgUnitPrice'})


scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)



optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

pickle.dump(kmeans, open( 'model.pkl','wb' ) )





# Analyze the clusters
cluster_summary = customer_data.groupby('Cluster').mean()
print(cluster_summary)

# Suggesting names for clusters based on their characteristics
for cluster in range(optimal_k):
    cluster_data = customer_data[customer_data['Cluster'] == cluster]
    print(f"\nCluster {cluster} Summary:")
    print(cluster_data.describe())

# Example names (adjust based on actual cluster characteristics)
cluster_names = {
    0: "High Value, Frequent Buyers",
    1: "Low Value, Occasional Buyers",
    2: "High Quantity, Low Value Buyers",
    3: "Average Buyers"
}

# Assign names to clusters
customer_data['ClusterName'] = customer_data['Cluster'].map(cluster_names)

# Visualize the clusters using seaborn pairplot with cluster names
sns.pairplot(customer_data, hue='ClusterName', palette='viridis')
plt.show()

