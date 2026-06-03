import pandas as pd
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    doc = nlp(text)

    tokens = []

    for token in doc:

        if not token.is_stop and not token.is_punct and not token.is_space:
            tokens.append(token.lemma_)

    return " ".join(tokens)