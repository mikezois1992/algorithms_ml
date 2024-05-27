import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px

from algorithms.pca import PCAHandler

st.title("Welcome to Michalis' Page")
st.title("Upload a File")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    # Display DataFrame
    st.write(df)

    tabs = st.tabs(["2D Visualization", "Classification", "Clustering"])

    with tabs[0]:
        st.header("PCA")

        scaled_df = StandardScaler().fit_transform(df.iloc[:, :-1])

        pca_handler = PCAHandler(scaled_df)
        pca_df = pca_handler.perform_pca()

        # Plot PCA
        pca_fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA Visualization")
        st.plotly_chart(pca_fig)
