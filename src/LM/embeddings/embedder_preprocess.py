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
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
def load_minilm_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
  
   
    print(f"Loading pretrained BERT model '{model_name}' from Hugging Face…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
   
    model.eval()
    print("BERT embedder loaded on device:", device)
    return tokenizer, model

# text -> embeddingy
def text_to_vector_minilm(text, tokenizer, model, device="cpu"):
    
    # Tokenize the sentence
    inputs = tokenizer(
          text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(device)
   

    for key in inputs:
        inputs[key] = inputs[key].to(device)

    # Pass through BERT
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  # shape: (hidden_size,)

    return cls_embedding.cpu().numpy() 


# rozdelenie dat na train test
def prepare_data_minilm(df, tokenizer, model, test_size=0.2, device="cpu", batch_size = 32):
    
    # Clean text
    df["Text"] = df["Text"].apply(clean_text)

    # Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["Text"], df["Category"], test_size=test_size, random_state=42, stratify=df["Category"]
    )

    # Convert to embeddings
    X_train_vec = text_to_vectors_minilm_batch(list(X_train_text), tokenizer, model, device=device, batch_size=batch_size)
    X_test_vec  = text_to_vectors_minilm_batch(list(X_test_text),  tokenizer, model, device=device, batch_size=batch_size)

    return X_train_vec, X_test_vec, y_train.reset_index(drop=True), y_test.reset_index(drop=True)





from tqdm import tqdm 

def text_to_vectors_minilm_batch(texts, tokenizer, model, device="cpu", batch_size=32):
   
    model.eval()
    all_embeddings = []

    # progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[i:i+batch_size]

        # tokenizacia
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)

        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Move to CPU & store
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)
