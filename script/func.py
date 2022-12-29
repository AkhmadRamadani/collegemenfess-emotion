# %% [markdown]
# ## Scrapping the tweets between 5th of December - 11st of December
# 
# Uncoment below line to re-scrap

# %%
# import twint
# import nest_asyncio

# nest_asyncio.apply()

# c = twint.Config()
# c.Username = "collegemenfess"
# c.Since = "2022-12-05"
# c.Until = "2022-12-11"
# c.Store_csv = True
# c.Output = "collegemenfess.csv"
# c.Limit = 5000
# c.Pandas = True

# twint.run.Search(c)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models import Word2Vec
# import MultinomialNB
from sklearn.naive_bayes import MultinomialNB
# import accuracy_score
from sklearn.metrics import accuracy_score
# import TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
# import LogisticRegression
from sklearn.linear_model import LogisticRegression

# download nltk data 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# load negatif and positif words from assets/negatif.txt and assets/positif.txt
# and store it in negatif and positif list
negatif = []
positif = []

with open('assets/negatif.txt', 'r') as f:
    for line in f:
        negatif.append(line.strip())

with open('assets/positif.txt', 'r') as f:
    for line in f:
        positif.append(line.strip())
        

# %%
# remove duplicate words in negatif and positif list
negatif = list(set(negatif))
positif = list(set(positif))

# %%
# import slang word from assets/slang.json
slang = pd.read_json('assets/slang.json', typ='series')
slang = slang.to_dict()


# %%
# add slang2.txt to slang
with open('assets/slang2.txt', 'r') as f:
    for line in f:
        line = line.strip().split(';')
        slang[line[0]] = line[1]


# %%
# function to convert slang word to formal word
def slang_to_formal(text):
    text = text.split()
    new_text = []
    for word in text:
        if word in slang:
            new_text.append(slang[word])
        else:
            new_text.append(word)
    return " ".join(new_text)

# %%
# acronym word from assets/acronym.txt to dictionary
acronym = {}
with open('assets/acronym.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split('=')
        acronym[key] = value



# %%
# function to convert acronym word to formal word
def acronym_to_formal(text):
    text = text.split()
    new_text = []
    for word in text:
        if word in acronym:
            new_text.append(acronym[word])
        else:
            new_text.append(word)
    return " ".join(new_text)

# %%
import re
def cleanTxt(text):
    text = re.sub(r'\[cm\]|\[CM\]|\[Cm\]|\[cM\]', '', text)
    text = re.sub(r'@\w+|#\w+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('\w+:\/\/\S+', '', text) # Removing hyperlink
    text = re.sub('[^a-zA-Z]', ' ', text)
    # lower case
    text = text.lower()
    return text

# stemming the tweet using Sastrawi 
def stemSentence(sentence):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    output   = stemmer.stem(sentence)
    return output

# remove stopword using Sastrawi
def removeStopWords(sentence):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    output = stopword.remove(sentence)
    return output

# classify the tweet using negatif and positif words
def classify(text):
    score = 0
    for word in text:
        if word in positif:
            score += 1
        elif word in negatif:
            score -= 1
    return score
# label the tweet based on actual_score
def labelTweet(score):
    if score > 0:
        return 'positif'
    elif score < 0:
        return 'negatif'
    else:
        return 'netral'

# predict the tweet
def predictTweet(tweet):
    tweet = cleanTxt(tweet)
    tweet = slang_to_formal(tweet)
    tweet = acronym_to_formal(tweet)
    tweet_baku = tweet
    tweet = stemSentence(tweet)
    tweet = removeStopWords(tweet)
    unigram = word_tokenize(tweet)
    score = classify(unigram)
    label = labelTweet(score)
    result = {
        'tweet_baku': tweet_baku,
        'processed_tweet': tweet,
        'score': score,
        'label': label
    }
    return result




