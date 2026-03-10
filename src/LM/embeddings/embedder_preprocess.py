import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import nltk
from tokenizers import Tokenizer
import torch
from transformers import AutoModel, AutoTokenizer
nltk.download('punkt', quiet=True)


EMBED_CACHE_VERSION = "minilm_embeddings_cache_v2.csv"



def load_data(path: str):
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
    return text.lower().strip()


# nacitanie predtrenovaneho word2vec modelu cez api
def load_word2vec_model():
    print("Loading pretrained Word2Vec model from gensim…")
    model = api.load("word2vec-google-news-300")
    print("Model loaded")
    return model


# text -> embeddingy - cele vety
def text_to_vector(text, model):
    words = word_tokenize(text)
    valid_words = [w for w in words if w in model.key_to_index] # iba slova ktore existuju v slovniku Word2Vec
    if not valid_words:
        return np.zeros(model.vector_size) # slova ktore word2vec neobsahuje -> nulove vektory
    return np.mean(model[valid_words], axis=0) # priemer embeddingov slov -> embedding vety


# text -> embedding - len pre slova
def text_to_vector_word(text, model):
    words = word_tokenize(text)
    valid_words = [w for w in words if w in model.key_to_index] # iba slova ktore existuju v slovniku Word2Vec
    if not valid_words:
        return np.zeros(model.vector_size) # slova ktore word2vec neobsahuje -> nulove vektory
    return model[valid_words[0]]



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


""" MiniLM embedder"""

# nacitanie predtrenovaneho word2vec modelu
def load_minilm_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
   
    print(f"Loading pretrained MiniLM model '{model_name}' from Hugging Face…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
   
   
    print("Minilm embedder loaded on device:", device)
    return tokenizer, model

# text -> embeddingy
def text_to_vector_minilm(text, tokenizer, model, device="cpu", cache = None):

    # cache - embeddingy ulozene v csv subore, aby sa nemuseli stale generovat nanovo
    if cache is not None:
        row = cache[(cache["text"] == text) & (cache["version"] == EMBED_CACHE_VERSION)]
        if not row.empty:
            return np.array(json.loads(row.iloc[0]['embedding']))


    
    # tokenizacia vety
    inputs = tokenizer(
          text,
        return_tensors="pt",
        truncation=True,
        max_length=250,
        padding="max_length"
    ).to(device)
   

    for key in inputs:
        inputs[key] = inputs[key].to(device)

    # forward pass cez miniLM
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  

   # da sa pouzit aj mean pooling

    vector = cls_embedding.cpu().numpy()

    #save to cache (ak to z nej nebolo)
    if cache is not None:
        new_row = {
            "text": text,
            "embedding": json.dumps(cls_embedding.tolist()),
            "version": EMBED_CACHE_VERSION
        }
        cache.loc[len(cache)] = new_row
        

    return vector








def prepare_data_minilm_simple(df, tokenizer, model, test_size=0.2, device="cpu"):
    
    # Clean text
    df["Text"] = df["Text"].apply(clean_text)

    # Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["Text"], df["Category"], test_size=test_size, random_state=42, stratify=df["Category"]
    )

    # load saved cache
    cache = load_embedding_cache()

    # Convert to embeddings
    # X_train_vec = text_to_vectors_minilm_batch(list(X_train_text), tokenizer, model, device=device, batch_size=batch_size)
    # X_test_vec  = text_to_vectors_minilm_batch(list(X_test_text),  tokenizer, model, device=device, batch_size=batch_size)

    # konverzia textu na embeddingy
    X_train_vec = np.array([text_to_vector_minilm(text, tokenizer, model, device, cache).flatten()
                        for text in X_train_text])
    X_test_vec  = np.array([text_to_vector_minilm(text, tokenizer, model, device, cache).flatten()
                        for text in X_test_text])


    cache.to_csv(EMBED_CACHE_VERSION, index=False)
    print(f"Saved cache with {len(cache)} entries.")

    return X_train_vec, X_test_vec, y_train.reset_index(drop=True), y_test.reset_index(drop=True)





def load_embedding_cache():
    try:
        cache = pd.read_csv(EMBED_CACHE_VERSION)
        print("Loaded embedding cache.")
    except FileNotFoundError:
        cache = pd.DataFrame(columns=["text", "embedding", 'version'])
        print("No cache found. Creating a new one.")
    return cache