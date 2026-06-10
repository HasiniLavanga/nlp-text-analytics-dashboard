import pandas as pd
import re
import spacy

# No external model required
nlp = spacy.blank("en")

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

        if not token.is_stop and not token.is_punct:
            tokens.append(token.text)

    return " ".join(tokens)
