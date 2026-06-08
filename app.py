import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocessing import clean_text
from src.topic_model import get_topics
from src.sentiment import get_sentiment
from src.ner import extract_entities

st.set_page_config(
    page_title="NLP Text Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NLP Text Analytics Dashboard")

st.header("Day 5 - Named Entity Recognition")

try:

    df = pd.read_csv(
        "Womens Clothing E-Commerce Reviews.csv"
    )

    text_column = "Review Text"

    df = df.dropna(subset=[text_column])

    sample_df = df.head(500)

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

    st.success("Data Processing Completed ✅")

    # Sentiment Chart

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

    # Topic Modeling

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

    # NER

    st.subheader("Named Entity Recognition")

    sample_text = sample_df[
        text_column
    ].iloc[0]

    entities = extract_entities(
        sample_text
    )

    if entities:

        entity_df = pd.DataFrame(
            entities
        )

        st.dataframe(entity_df)

    else:

        st.info(
            "No entities found in sample review."
        )

except Exception as e:
    st.error(str(e))
