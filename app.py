import streamlit as st
import pandas as pd

from src.preprocessing import clean_text
from src.topic_model import get_topics

st.set_page_config(
    page_title="NLP Text Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NLP Text Analytics Dashboard")

st.header("Day 3 - Topic Modeling")

try:

    df = pd.read_csv(
        "Womens Clothing E-Commerce Reviews.csv"
    )

    text_column = "Review Text"

    df = df.dropna(subset=[text_column])

    sample_df = df.head(500)

    with st.spinner("Cleaning Text..."):

        sample_df["Cleaned_Text"] = sample_df[text_column].apply(
            clean_text
        )

    st.success("Text Preprocessing Completed ✅")

    st.subheader("Sample Cleaned Reviews")

    st.dataframe(
        sample_df[
            [text_column, "Cleaned_Text"]
        ].head()
    )

    st.subheader("LDA Topic Modeling")

    topics = get_topics(
        sample_df["Cleaned_Text"]
    )

    for topic in topics:

        st.write(
            f"**{topic['Topic']}**"
        )

        st.write(
            topic["Keywords"]
        )

        st.write("---")

except Exception as e:
    st.error(str(e))
