import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px

from algorithms.eda import EDAHandler
from algorithms.pca import PCAHandler
from algorithms.tsne import TSNEHandler

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

        st.header("t-SNE")

        tsne_handler = TSNEHandler(scaled_df)
        tsne_df = tsne_handler.perform_tsne()

        # Plot t-SNE
        tsne_fig = px.scatter(tsne_df, x="t-SNE1", y="t-SNE2", title="t-SNE Visualization")
        st.plotly_chart(tsne_fig)

        st.header("EDA")

        # Perform EDA
        eda_handler = EDAHandler(df)
        eda_handler.perform_eda()