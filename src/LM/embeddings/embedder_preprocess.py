import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)



def load_data(path: str):
    """Loads a labeled text dataset from CSV and prints sample rows."""
    df = pd.read_csv(path, names=["Category", "Text"], header=None)
    print("Dataset shape:", df.shape)

    print("\nCategory samples:")
    df_sample = df.groupby('Category', group_keys=False).head(1).sort_values(by='Category')

    for _, row in df_sample.iterrows():
        category = row['Category']
        description = row['Text'][:400]
        print(f"\n{category}:")
        print(f"  {description}...")

    pd.set_option('display.max_colwidth', 50)
    return df



def clean_text(text):
    
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# nacitanie predtrenovaneho word2vec modelu
def load_word2vec_model():
    print("Loading pretrained Word2Vec model from gensim…")
    model = api.load("word2vec-google-news-300")  # ~1.6 GB, loads once
    print("Model loaded")
    return model


# text -> embeddingy
def text_to_vector(text, model):
    words = word_tokenize(text.lower())
    valid_words = [w for w in words if w in model.key_to_index]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean(model[valid_words], axis=0)


# rozdelenie dat na train test
def prepare_data_word2vec(df, test_size=0.2):
    # clean text
    df["Text"] = df["Text"].apply(clean_text)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        df["Text"],
        df["Category"],
        test_size=test_size,
        random_state=42,
        stratify=df["Category"]
    )

    # nacitanie modelu
    model = load_word2vec_model()

    # konverzia textu na vektory
    X_train_vec = np.array([text_to_vector(text, model) for text in X_train])
    X_test_vec = np.array([text_to_vector(text, model) for text in X_test])

    return X_train_vec, X_test_vec, y_train, y_test, model


