from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def get_topics(texts, n_topics=5):

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words="english"
    )

    dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

    lda.fit(dtm)

    words = vectorizer.get_feature_names_out()

    topics = []

    for idx, topic in enumerate(lda.components_):

        top_words = [
            words[i]
            for i in topic.argsort()[-10:]
        ]

        topics.append({
            "Topic": f"Topic {idx+1}",
            "Keywords": ", ".join(top_words)
        })

    return topics