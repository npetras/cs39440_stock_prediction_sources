"""
Module for pre-processing and cleaning data (specifically stock market), but is also suitable 
for other data. The pre-processing is to prepare text data for sentiment analysis.
"""
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import treebank

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
    return text.translate(str.maketrans('', '', string.punctuation))
    
def remove_stopwords(text):
    """
    """
    stop_words = stopwords.words('english')
    words = word_tokenize(text)
    # regex for stopwords that should not be removed from text, since they have negative sentiment
    whitelist = [".*n't", 'not', 'no']
    modified_stop_words = []
    processed_text_list = []

    # create a new list of stop words without those on the white list
    for stop_word in stop_words:
        if re.match(whitelist[0], stop_word) or re.match(whitelist[1], stop_word) or re.match(whitelist[2], stop_word):
            continue
        else:
            modified_stop_words.append(stop_word)
    
    processed_text_list = [word for word in words if (word not in modified_stop_words)]

    return treebank.TreebankWordDetokenizer().detokenize(processed_text_list)

def lemmatization(text):
    """
    """
    pass

def apply_all(text):
    processed_text = to_lowercase(text)

# read in file

# reduce it to ten records