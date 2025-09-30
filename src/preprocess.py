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
    text = re.sub(r"[^a-z\s]", "", text)        #remove all that are not lowercase leeters or whitespace

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

    vectorizer = TfidfVectorizer(max_features=5000)     #max_features = 5000 - limit vocabulary to 5000 most used words
    X_train_vec = vectorizer.fit_transform(X_train)     #vectorizer applied only for train data (vocabulary, weights, transform data)
    X_test_vec = vectorizer.transform(X_test)           #transform data with the vocabulary and weights learned

    #return vectorized features, category labels and vectorizer
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer