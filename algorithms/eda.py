import plotly.express as px
import streamlit as st

class EDAHandler:
    def __init__(self, df):
        self.df = df

    def perform_eda(self):
        for i in range(len(self.df.columns) - 1):  # Exclude the target variable
            for j in range(i + 1, len(self.df.columns) - 1):
                eda_fig = px.scatter(self.df, x=self.df.columns[i], y=self.df.columns[j],
                                     title=f"EDA: {self.df.columns[i]} vs {self.df.columns[j]}")
                st.plotly_chart(eda_fig)