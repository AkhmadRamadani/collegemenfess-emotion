import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import script.func as func
import pickle

from gensim.models import Word2Vec

# df = pd.read_csv('data/collegemenfess_cluster.csv')

# import kmeans_model
kmeans = pickle.load(open('models/kmeans.sav', 'rb'))
svm = pickle.load(open('models/svm.sav', 'rb'))

# get vector of tweet
def get_vector(tweet):
    vector = np.zeros(100)
    count = 0
    for word in tweet.split():
        try:
            vector += Word2Vec.load('models/word2vec.model').wv[word]
            count += 1
        except:
            pass
    vector = vector / count
    return vector


# predict the test_data using kmeans
def predict_label(tweet):
    tweet = func.cleanTxt(tweet)
    tweet = func.slang_to_formal(tweet)
    tweet = func.acronym_to_formal(tweet)
    real_tweet = tweet
    tweet = func.stemSentence(tweet)
    tweet = func.removeStopWords(tweet)
    vector = get_vector(tweet)
    reshaped_vector = vector.reshape(1, -1)
    label = kmeans.predict(reshaped_vector)
    return {
        'label': label.tolist()[0],
        'processed_tweet': tweet,
        'real_tweet': real_tweet
    }


# predict the test_data using svm
def predict_label_svm(tweet):
    real_tweet = tweet
    tweet = func.cleanTxt(tweet)
    tweet = func.slang_to_formal(tweet)
    tweet = func.acronym_to_formal(tweet)
    tweet_baku = tweet
    tweet = func.stemSentence(tweet)
    tweet = func.removeStopWords(tweet)
    label = svm.predict([tweet])
    result = {
        'label': label.tolist()[0],
        'processed_tweet': tweet,
        'real_tweet': real_tweet,
        'tweet_baku': tweet_baku
    }
    return result
    

