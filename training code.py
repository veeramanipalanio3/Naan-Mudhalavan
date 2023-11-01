import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#from sklearn.metrics import accuracy_score, classification_report
#from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from nltk.tokenize import word_tokenize

# Load the dataset
data = pd.read_csv("spam_ham_dataset.csv")

# Preprocess the text data
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Removing punctuation and converting to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    
    # Removing stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back into a string
    preprocessed_text = " ".join(tokens)
    
    return preprocessed_text

data['text'] = data['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train an ensemble model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
ensemble_classifier = VotingClassifier(estimators=[('rf', rf_classifier)], voting='hard')

ensemble_classifier.fit(X_train, y_train)
print(X_train)
