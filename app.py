import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Bank Customer Segmentation", layout="wide")

st.title("Bank Customer Segmentation App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    customer_df = pd.read_csv(uploaded_file, encoding='latin1')

    # Initial data exploration
    st.subheader("Raw Data Sample")
    st.write(customer_df.head())

    # Handle missing values
    customer_df['CustAccountBalance'] = customer_df['CustAccountBalance'].fillna(customer_df['CustAccountBalance'].median())
    customer_df['TransactionAmount (INR)'] = customer_df['TransactionAmount (INR)'].fillna(customer_df['TransactionAmount (INR)'].median())

    # Convert DOB to age
    customer_df['CustomerDOB'] = pd.to_datetime(customer_df['CustomerDOB'], errors='coerce')
    customer_df['Age'] = 2025 - customer_df['CustomerDOB'].dt.year
    customer_df.drop('CustomerDOB', axis=1, inplace=True)

    # Encode categorical variables
    customer_df['GenderEncoded'] = LabelEncoder().fit_transform(customer_df['CustGender'].astype(str))
    customer_df['LocationEncoded'] = LabelEncoder().fit_transform(customer_df['CustLocation'].astype(str))

    st.subheader("Age Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(customer_df['Age'], bins=30, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Account Balance vs Transaction Amount")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='CustAccountBalance', y='TransactionAmount (INR)', data=customer_df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    numeric_df = customer_df.select_dtypes(include=['number'])
    fig3, ax3 = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Clustering (KMeans)")
    features = customer_df[['Age', 'CustAccountBalance', 'TransactionAmount (INR)', 'GenderEncoded', 'LocationEncoded']]
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    features_imputed = imputer.fit_transform(features)
    features_scaled = scaler.fit_transform(features_imputed)

    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)

    fig4, ax4 = plt.subplots()
    ax4.plot(K, inertia, marker='o')
    ax4.set_title("Elbow Method")
    ax4.set_xlabel("Number of Clusters (K)")
    ax4.set_ylabel("Inertia")
    st.pyplot(fig4)

    optimal_k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=4)

    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
    customer_df['Cluster'] = kmeans_final.fit_predict(features_scaled)

    st.subheader("Clustered Data Sample")
    st.write(customer_df[['Age', 'CustAccountBalance', 'TransactionAmount (INR)', 'Cluster']].head())

    st.subheader("Cluster Visualization")
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=customer_df, x='CustAccountBalance', y='TransactionAmount (INR)', hue='Cluster', palette='tab10', ax=ax5)
    st.pyplot(fig5)

else:
    st.info("Please upload a dataset to begin.")
