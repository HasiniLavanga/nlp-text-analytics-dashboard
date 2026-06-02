import streamlit as st
import pandas as pd

# Page Config
st.set_page_config(
    page_title="NLP Text Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 NLP Text Analytics Dashboard")
st.write("Day 1 - Dataset Overview")

try:
    # Load Dataset
    df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

    st.success("Dataset Loaded Successfully ✅")

    # Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Columns
    st.subheader("Column Names")
    st.write(df.columns.tolist())

    # Missing Values
    st.subheader("Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    st.dataframe(missing)

except Exception as e:
    st.error(f"Error: {e}")