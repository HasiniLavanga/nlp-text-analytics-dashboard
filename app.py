import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import clean_text
from topic_model import get_topics
from sentiment import get_sentiment
from ner import extract_entities

st.set_page_config(
    page_title="NLP Text Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NLP Text Analytics Dashboard")
st.markdown("### Customer Review Analytics using NLP")

try:

    # Load Dataset
    df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

    text_column = "Review Text"

    df = df.dropna(subset=[text_column])

    sample_df = df.head(1000).copy()

    # Text Preprocessing
    sample_df["Cleaned_Text"] = sample_df[text_column].apply(clean_text)

    # Sentiment Analysis
    sentiments = sample_df["Cleaned_Text"].apply(get_sentiment)

    sample_df["Sentiment"] = sentiments.apply(lambda x: x[0])
    sample_df["Score"] = sentiments.apply(lambda x: x[1])

    st.success("Data Processing Completed Successfully ✅")

    # Metrics
    st.header("📈 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Reviews", len(sample_df))

    with col2:
        st.metric(
            "Positive Reviews",
            len(sample_df[sample_df["Sentiment"] == "Positive"])
        )

    with col3:
        st.metric(
            "Negative Reviews",
            len(sample_df[sample_df["Sentiment"] == "Negative"])
        )

    # Sentiment Chart
    st.header("😊 Sentiment Analysis")

    sentiment_counts = (
        sample_df["Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        sentiment_counts,
        values="Count",
        names="Sentiment",
        title="Sentiment Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Topic Modeling
    st.header("📌 Topic Modeling")

    topics = get_topics(sample_df["Cleaned_Text"])

    for topic in topics:
        st.subheader(topic["Topic"])
        st.write(topic["Keywords"])

    # Search Reviews
    st.header("🔍 Search Reviews")

    search_text = st.text_input("Enter keyword")

    if search_text:

        filtered = sample_df[
            sample_df[text_column].str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

        st.dataframe(
            filtered[
                [text_column, "Sentiment"]
            ]
        )

    # Named Entity Recognition
    st.header("🏷 Named Entity Recognition")

    sample_review = sample_df[text_column].iloc[0]

    entities = extract_entities(sample_review)

    if entities:

        entity_df = pd.DataFrame(entities)

        st.dataframe(entity_df)

    else:

        st.info("No entities found.")

    # Dataset Preview
    st.header("📄 Dataset Preview")

    st.dataframe(sample_df.head(20))

except Exception as e:

    st.error(f"Error: {str(e)}")
