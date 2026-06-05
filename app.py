import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocessing import clean_text
from src.topic_model import get_topics
from src.sentiment import get_sentiment

st.set_page_config(
    page_title="NLP Text Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NLP Text Analytics Dashboard")

st.header("Day 4 - Sentiment Analysis")

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

    sentiments = sample_df["Cleaned_Text"].apply(
        get_sentiment
    )

    sample_df["Sentiment"] = sentiments.apply(
        lambda x: x[0]
    )

    sample_df["Score"] = sentiments.apply(
        lambda x: x[1]
    )

    st.success("Sentiment Analysis Completed ✅")

    st.subheader("Sentiment Distribution")

    sentiment_counts = (
        sample_df["Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        title="Sentiment Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Sample Reviews")

    st.dataframe(
        sample_df[
            [
                text_column,
                "Sentiment",
                "Score"
            ]
        ].head(20)
    )

    st.subheader("Topic Modeling Results")

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
