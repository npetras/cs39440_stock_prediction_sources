import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, log_loss
import gensim
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

with open('alexa_reviews_clean.pkl', 'rb') as read_file:
    df = pickle.load(read_file)
print(df.head())

# Topic Modelling
# dictionary to count the words
count_dict_alexa = {}

for doc in df['new_reviews']:
    for word in doc.split():
        if word in count_dict_alexa.keys():
            count_dict_alexa[word] += 1
        else:
            count_dict_alexa[word] = 1
    
    for key, value in sorted(count_dict_alexa.items(), key=lambda item: item[1]):
        print("%s: %s" % (key, value))

# remove words that occur in-frequently
low_value = 10
bad_words = [key for key in count_dict_alexa.keys() if count_dict_alexa[key] < low_value]

corpus = [doc.split() for doc in df['new_reviews']]
clean_list = []
for document in corpus:
    clean_list.append([word for word in document if word not in bad_words])

print(clean_list)

# inputs for LDA
corpora_dict = corpora.Dictionary(clean_list)
corpus = [corpora_dict.doc2bow(line) for line in clean_list]

# train LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=corpora_dict,
    random_state=100,
    num_topics=3,
    passes=5,
    per_word_topics=True
)

print(lda_model.print_topics(-1))