import numpy as np 
import pandas as pd
pd.set_option('display.max_colwidth', None)
from time import time
import re
import string
# operatating system library - using OS dependent functionality
import os
import emoji
# pretty print
from pprint import pprint
import collections

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib

import gensim

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import warnings
# warnings.filterwarnings('ignore')

np.random.seed(37)

df = pd.read_csv('./Tweets.csv')
df = df.reindex(np.random.permutation(df.index))
df = df[['text', 'airline_sentiment']]

sns.catplot(x="airline_sentiment", data=df, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")
plt.show();