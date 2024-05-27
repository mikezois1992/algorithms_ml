import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
import plotly.express as px

from algorithms.dbscan import DBSCANHandler
from algorithms.eda import EDAHandler
from algorithms.kmeans import KMeansHandler
from algorithms.logistic_regression import LogisticRegressionClassifier
from algorithms.pca import PCAHandler
from algorithms.svm import SVMClassifier
from algorithms.tsne import TSNEHandler

import random

st.title("Welcome to Michalis' Page")
st.title("Upload a File")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    # Display DataFrame
    st.write(df)

    tabs = st.tabs(["2D Visualization", "Classification", "Clustering"])

    with tabs[0]:
        st.header("Clustering Algorithms")
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

    with tabs[1]:
        st.header("Classification Algorithms")

        # Assuming the last column is the target variable
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Binarize the output
        y_bin = label_binarize(y, classes=list(y.unique()))
        n_classes = y_bin.shape[1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)


        def plot_classification_results(model, X_test, y_test, y_pred, y_score, model_name):
            # Display accuracy for each class
            st.write(f"### {model_name}")
            for i in range(n_classes):
                accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
                st.write(f"Accuracy for class {i}: {accuracy:.2f}")

            # Plot Confusion Matrix for each class
            for i in range(n_classes):
                cm = confusion_matrix(y_test[:, i], y_pred[:, i])
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Class {i}', f'Not Class {i}'])
                disp.plot(ax=ax)
                st.write(f"Confusion Matrix for class {i} ({model_name})")
                st.pyplot(fig)

            # Plot ROC Curve for each class
            fig, ax = plt.subplots()
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'ROC curve for class {i} (area = {roc_auc:.2f})')

            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Receiver Operating Characteristic ({model_name})')
            ax.legend(loc="lower right")
            st.pyplot(fig)

        # Logistic Regression
        log_reg = LogisticRegressionClassifier()
        log_reg.fit(X_train, y_train)
        y_pred_log_reg = log_reg.predict(X_test)
        y_score_log_reg = log_reg.decision_function(X_test)
        plot_classification_results(log_reg, X_test, y_test, y_pred_log_reg, y_score_log_reg, "Logistic Regression")

        # SVM
        svm = SVMClassifier()
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        y_score_svm = svm.decision_function(X_test)
        plot_classification_results(svm, X_test, y_test, y_pred_svm, y_score_svm, "SVM")

    with tabs[2]:
        st.header("K-means")

        num_clusters = st.number_input("Number of Clusters for K-means", min_value=1, step=1)

        kmeans_handler = KMeansHandler(scaled_df, num_clusters)
        kmeans_result = kmeans_handler.perform_kmeans()

        # Assign colors to clusters
        cluster_colors = {}
        for cluster in set(kmeans_result):
            cluster_colors[cluster] = f'#{random.randint(0, 0xFFFFFF):06x}'  # Generate random color

        # Plot K-means result
        kmeans_fig = px.scatter(x=pca_df["PC1"], y=pca_df["PC2"], color=kmeans_result.astype(str),
                                color_discrete_map=cluster_colors, title="K-means Clustering")
        st.plotly_chart(kmeans_fig)

        st.header("DBSCAN")

        eps = st.number_input("Epsilon for DBSCAN", min_value=0.1, step=0.1)
        min_samples = st.number_input("Min Samples for DBSCAN", min_value=1, step=1)

        dbscan_handler = DBSCANHandler(scaled_df, eps, min_samples)
        dbscan_result = dbscan_handler.perform_dbscan()

        # Plot DBSCAN result
        dbscan_fig = px.scatter(x=pca_df["PC1"], y=pca_df["PC2"], color=dbscan_result.astype(str),
                                color_discrete_map=cluster_colors, title="DBSCAN Clustering")
        st.plotly_chart(dbscan_fig)