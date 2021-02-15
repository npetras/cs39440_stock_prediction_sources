"""
Module for pre-processing and cleaning data (specifically stock market), but is also suitable 
for other data. The pre-processing is to prepare text data for sentiment analysis.
"""
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import treebank

import spacy

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

def remove_punct(text):
    """
    Returns the string 'text' with its punctuation removed, this function expects
    'text' to be a string.
    """
    punctuation = string.punctuation
    # text without punctuation (punctuation has been replaced by spaces)
    processed_text = text.translate(str.maketrans(punctuation, len(punctuation) * ' '))
    # remove double, triple spaces with a single space
    processed_text = re.sub('\s+', ' ', processed_text)
    # remove dangling space
    # s at the end of the string
    return re.sub('\s+$', '', processed_text)
    
def remove_stopwords(text):
    """
    Returns the 'text' with all its stopwords removed. The list of stopwords used 
    is nltk's list of english stopwords.
    """
    stop_words = stopwords.words('english')
    words = word_tokenize(text)    
    processed_text_list = [word for word in words if (word not in stop_words)]
    return treebank.TreebankWordDetokenizer().detokenize(processed_text_list)

def lemmatize_text(text):
    """
    Returns the text with lemmatization applied to it -- each word in converted 
    to its base dictionary word (lemma).
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    return treebank.TreebankWordDetokenizer().detokenize(tokens)


def apply_all(text):
    """
    Returns the 'text' with all of the above pre-processing steps applied to it, in the correct order: 
    convert text to lowercase, remove pucntuation, remove stop words, and applies lemmatization to 
    the text. 
    """
    processed_text = to_lowercase(text)
    processed_text = remove_punct(processed_text)
    processed_text = remove_stopwords(processed_text)
    return lemmatize_text(processed_text)

# read in file

# reduce it to ten records