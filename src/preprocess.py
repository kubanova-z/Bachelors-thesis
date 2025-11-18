import re   #module for text cleaning
from sklearn.model_selection import train_test_split    #function for splitting data into test / train sets
from sklearn.feature_extraction.text import TfidfVectorizer     #vectorizer for extraction


""" 
Vectorizer: text -> numerical data
- feature extraction (converting raw text into matrix of numerical feature vectors)
1. vocabulary learning + feature creation (each unique word -> feature ), mappings
2. transform (each descrption -> vector, each number represents the importance of a word), numerical value = TD-IDF score (term frequency (how often) * inverse document frequency (how rare))
output: sparse matrix (rows -> descriptions, columns -> 5000 unique words, values -> TF-IDF score)
 """

def clean_text(text):
    #check - convert non string data to string
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()     #lowercase
    
    #remove punctation, numbers, special symbols
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        #remove all that are not lowercase leeters or whitespace
    
    #remove numbers
    text = re.sub(r"\d+", " ", text)

    text = text.replace("_", " ")


    #convert multiple spaces into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


#dataFrame, 20% of data for the test split
def prepare_data(df, test_size = 0.2):

    #cleaning of the text
    df["Text"] = df["Text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Text"],     #features
        df["Category"],     #labels
        test_size=test_size,    #20% for test
        random_state=42,        
        stratify=df["Category"]     #proportion of each category is even in test and train data
    )

    #chnage for smaller dataset
    vectorizer = TfidfVectorizer(max_features=3000,
                                 ngram_range=(1, 2),
                                 min_df=3,
                                 max_df=0.9,
                                 sublinear_tf=True)     #max_features = 5000 - limit vocabulary to 5000 most used words
    X_train_vec = vectorizer.fit_transform(X_train)     #vectorizer applied only for train data (vocabulary, weights, transform data)
    X_test_vec = vectorizer.transform(X_test)           #transform data with the vocabulary and weights learned

    #return vectorized features, category labels and vectorizer
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


import pandas as pd

def print_vectorized_sample():
    sample_text = (
        "Lovely Arts Collection Hand Embroidered Cotton Thread Skeins for Craft Projects, Multicolour (Pack of 25) -LAC97 Package includes 25 skeins,"
        " two each of 12 colors. Each skein is 8 yards long. "
        "This quality embroidery floss can be used in a multitude of craft projects, including needlecraft, friendship bracelets, stringing beads, and more. Non-toxic."
        )
    
    print("\n" + "="*50)
    print("Text preprocess")
    print("="*50)
    
    
    print("--- 1. Input Sample ---")
    print(sample_text)
    print("-" * 30)

    cleaned_text = clean_text(sample_text)
    print("--- 2. Cleaned Text (Lowercase, without interpunction / numbers) ---")
    print(cleaned_text)
    print(f"Count of tokens in cleaned text: {len(cleaned_text.split())}")
    print("-" * 30)

    vectorizer = TfidfVectorizer(max_features=5000)
    vectorized_output = vectorizer.fit_transform([cleaned_text])

    print("--- Vectorization Sample ---")
    print(f"Shape of vectorized output (documents, features): {vectorized_output.shape}")
    print("-" * 30)

    print(" Sample has 42 words.\n"
      " Coords (0 - we have just 1 sample, n - position of the word)\n"
      " Value (TF-IDF score - the smaller the more important)")

    sample_vector = vectorized_output[0]
    
    print("Vector for the sample text (sparse format):")
    print(sample_vector)
    
    print("\nSame vector in dense format (array of TF-IDF scores):")
    print(sample_vector.toarray())
    print("-" * 30)
    
    feature_names = vectorizer.get_feature_names_out()

    print("Features and TF-IDF scores for the sample text:")
    for col, score in zip(sample_vector.indices, sample_vector.data):
        print(f"  - Word: '{feature_names[col]}', Score: {score:.4f}")

   