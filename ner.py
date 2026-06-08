import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):

    doc = nlp(str(text))

    entities = []

    for ent in doc.ents:
        entities.append(
            {
                "Entity": ent.text,
                "Label": ent.label_
            }
        )

    return entities