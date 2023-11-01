import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load the dataset
data = pd.read_csv('spam_ham_dataset.csv')
df = pd.DataFrame(data)

# Preprocess the text data

totaltokens = []
totalpunc=[]
totalstopwords=[]
totalstem=[]

def preprocess_text(text):
    # Tokenization
    global totaltokens
    tokens = word_tokenize(text)
    totaltokens.append(tokens)
    #print('AFTER TOKENIZATION:', totaltokens, "\n")
    
    
    # Removing punctuation and converting to lowercase
    global totalpunc
    tokens1 = [word.lower() for word in tokens if word.isalpha()]
    totalpunc.append(tokens1)
    #print('After Removing punctuation and converting to lowercase :', totalpunc, "\n")
   
    # Removing stopwords
    global totalstopwords
    stop_words = set(stopwords.words("english"))
    tokens2 = [word for word in tokens1 if word not in stop_words]
    totalstopwords.append(tokens2)
    #print('AFTER REMOVING STOPWORDS:', totalstopwords, "\n")
    
    # Stemming
    global totalstem
    stemmer = PorterStemmer()
    tokens3 = [stemmer.stem(word) for word in tokens2]
    totalstem.append(tokens3)
    #print('AFTER STEMMING:', totalstem, "\n")
   
    # Join tokens back into a string
    preprocessed_text = " ".join(tokens3)
    #print('AFTER JOINING:', preprocessed_text)
    
    return preprocessed_text

# Apply the preprocess_text function to the 'text' column
df['text'] = df['text'].head().apply(preprocess_text)

# Print the first few rows of the preprocessed text
print("AFTER PREPROCESSING:\n",df['text'].head())
print('AFTER TOKENIZATION:', totaltokens, "\n")
print('After Removing punctuation and converting to lowercase :', totalpunc, "\n")
print('AFTER REMOVING STOPWORDS:', totalstopwords, "\n")
print('AFTER STEMMING:', totalstem, "\n")
    
