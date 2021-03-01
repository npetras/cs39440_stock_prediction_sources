"""
Module for pre-processing and cleaning data (specifically stock market), but is also suitable 
for other data. The pre-processing is to prepare text data for sentiment analysis.
"""
import string
import re

from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.tokenize import treebank, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import spacy

from sklearn.feature_extraction.text import CountVectorizer 

# does not include % or $, since those are useful in terms of stocks
PUNCT_REGEX = """[!"#&\'()*\+,-.\/\\:;<=>?@\[\]^_`{}~\|]"""
DOUBLE_SPACE = '  '
SPACE = ' '
BLANK = ''

def to_lowercase(text):
    """
    Returns the 'text' as a lowercase string, by simply apply the lower() to the
    string.
    """
    return text.lower()

def sklearn_tokenize(text):
    """
    """
    tokenizer = CountVectorizer().build_tokenizer()
    return tokenizer(text)

def detokenize(tokenized_text):
    detokenizer = treebank.TreebankWordDetokenizer()
    return detokenizer.detokenize(tokenized_text)
    
def remove_stopwords(tokenized_text):
    """
    Returns the 'text' with all its stopwords removed. The list of stopwords used 
    is nltk's list of english stopwords.
    """
    stop_words = stopwords.words('english')
    return [token.lower() for token in tokenized_text if (token.lower() not in stop_words)]
 

def porter_stem(tokenized_text):
    stemmer = PorterStemmer()
    return [stemmer.stem(token.lower()) for token in tokenized_text]

def tag_with_pos(pos_tag):
    if pos_tag.startswith('N'):
        simplified_pos_tag = wordnet.NOUN
    elif pos_tag.startswith('V'):
        simplified_pos_tag = wordnet.VERB
    elif pos_tag.startswith('J'):
        simplified_pos_tag = wordnet.ADJ
    elif pos_tag.startswith('R'):
        simplified_pos_tag = wordnet.ADV
    else:
        simplified_pos_tag = None
    return simplified_pos_tag

def word_net_lemmatize(tokens):
    lemmatized_tokens = []
    wordnet_lemmatizer = WordNetLemmatizer()

    for token, pos in pos_tag(tokens):
        simplified_pos = tag_with_pos(pos)
        if simplified_pos != None:
            lemmatized_token = wordnet_lemmatizer.lemmatize(token, simplified_pos)
            lemmatized_tokens.append(lemmatized_token)
        else:
            lemmatized_tokens.append(token)

    return lemmatized_tokens


# read in file

# reduce it to ten records
