import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Title
st.title("WebApp Using Breast Cancer Prediction Using Machine Learning")

# Image
st.image("strea.png", width=500)

# Load the breast cancer dataset from scikit-learn
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target

# Information about the dataset
st.title("Case Study on The Breast Cancer Dataset")
st.write("The target variable indicates whether the tumor is malignant (1) or benign (0).")
st.write("Shape of the dataset:", df.shape)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Home", "Prediction Report", "Feature Correlation", "Dataset Overview"])

if menu == "Home":
    st.image("Breastcancers.png", width=550)
    st.header("Tabular Data of the Breast Cancer Dataset")
    if st.checkbox("Show Tabular Data"):
        st.table(df.head(10))

    # Statistical summary of the DataFrame
    st.header("Statistical Summary of the DataFrame")
    if st.checkbox("Show Statistics"):
        st.table(df.describe())

    # Correlation Graph
    st.header("Correlation Graph")
    if st.checkbox("Show Correlation Graph"):
        # Filter numeric columns only for correlation
        numeric_df = df.select_dtypes(include=[float, int])
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
        st.pyplot(fig)

    # Input fields for each feature
    st.title("Input Features for Prediction")
    clump_thickness = st.number_input("Clump Thickness", min_value=0.0, max_value=10.0, value=5.0)
    uniformity_of_cell_size = st.number_input("Uniformity of Cell Size", min_value=0.0, max_value=10.0, value=5.0)
    uniformity_of_cell_shape = st.number_input("Uniformity of Cell Shape", min_value=0.0, max_value=10.0, value=5.0)
    marginal_adhesion = st.number_input("Marginal Adhesion", min_value=0.0, max_value=10.0, value=5.0)
    single_epithelial_cell_size = st.number_input("Single Epithelial Cell Size", min_value=0.0, max_value=10.0, value=5.0)
    bare_nuclei = st.number_input("Bare Nuclei", min_value=0.0, max_value=10.0, value=5.0)
    bland_chromatin = st.number_input("Bland Chromatin", min_value=0.0, max_value=10.0, value=5.0)
    normal_nucleoli = st.number_input("Normal Nucleoli", min_value=0.0, max_value=10.0, value=5.0)
    mitoses = st.number_input("Mitoses", min_value=0.0, max_value=10.0, value=5.0)

    # Collect inputs into a numpy array for prediction
    user_data = np.array([[clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape, marginal_adhesion,
                           single_epithelial_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses]])

    # Train a model (this should ideally be done outside of the Streamlit app and saved)
    X = df.drop(columns=['target'])
    y = df['target']
    model = RandomForestClassifier()
    model.fit(X, y)

    # When the Predict button is clicked, make a prediction
    if st.button('Predict'):
        prediction = model.predict(user_data)
        result = 'Malignant' if prediction[0] == 1 else 'Benign'
        st.success(f"The tumor is predicted to be: **{result}**")

elif menu == 'Feature Correlation':
    st.write("### Feature Correlation Matrix")
    
    # Compute correlation matrix
    corr_matrix = df.corr()

    # Plot the correlation heatmap using Seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    
    # Display the heatmap
    st.pyplot(fig)

elif menu == 'Dataset Overview':
    st.write("### Breast Cancer Dataset")
    st.dataframe(df.head())
    st.write("### Dataset Description")
    st.write(df.describe())
