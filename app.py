import streamlit as st
import pandas as pd

from src.preprocessing import clean_text

st.set_page_config(
    page_title="NLP Text Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NLP Text Analytics Dashboard")

st.header("Day 2 - Text Preprocessing")

try:

    df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

    st.success("Dataset Loaded Successfully ✅")

    # Find review column
    text_column = "Review Text"

    # Remove missing reviews
    df = df.dropna(subset=[text_column])

    st.write("Rows after removing missing values:", len(df))

    # Sample size for faster processing
    sample_df = df.head(200)

    with st.spinner("Cleaning text..."):

        sample_df["Cleaned_Text"] = sample_df[text_column].apply(clean_text)

    st.subheader("Original Review")

    st.write(sample_df[text_column].iloc[0])

    st.subheader("Cleaned Review")

    st.write(sample_df["Cleaned_Text"].iloc[0])

    st.subheader("Dataset Preview")

    st.dataframe(
        sample_df[
            [text_column, "Cleaned_Text"]
        ].head(10)
    )

except Exception as e:
    st.error(str(e))
