import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# does not include % or $, since those are useful in terms of stocks
PUNCT_REGEX = """[!"#&\'()*\+,-.\/\\:;<=>?@\[\]^_`{}~\|]"""
DOUBLE_SPACE = '  '
SPACE = ' '
BLANK = ''

def to_lowercase(text):
    return text.lower()

def remove_punct(text):
    """
    
    """
    return text.translate(str.maketrans('', '', string.punctuation))
    
def remove_stopwords(text):
    """

    """
    stop_words = stopwords.words('english')
    words = text.word_tokenize()
    # regex for stopwords that should not be removed from text, since they have negative sentiment
    whitelist = [".*n't", 'not', 'no']
    modified_stop_words = []
    processed_text = []

    # create a new list of stop words without those on the white list
    for stop_word in stop_words:
        if re.match(whitelist[0], stop_word) or re.match(whitelist[1], stop_word) or re.match(whitelist[2], stop_word):
            continue
        else:
            modified_stop_words.append(stop_word)
    
    processed_text = [word for word in words if (word not in modified_stop_words)]

    return processed_text

def lemmatization(text):
    """
    """
    pass

# read in file

# reduce it to ten records