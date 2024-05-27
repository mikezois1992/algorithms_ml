import streamlit as st
import pandas as pd

st.title("Welcome to Michalis' Page")
st.title("Upload a File")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    # Display DataFrame
    st.write(df)