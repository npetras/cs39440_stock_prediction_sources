"""
Contains all functions used to clean the data/remove noise from the data.
"""

import sklearn.feature_extraction.text as feature_extraction
import nltk
from nltk.tokenize import treebank
import nltk.stem.wordnet
import nltk.stem.porter
# pylint: disable=E0611
# issue with nltk, or bad error from pylint
import nltk.corpus.reader.wordnet as wordnet_corpus


def sklearn_tokenize(text):
    """
    Runs the sklearn tokenizer on the text provided.
    :param text: string of text that needs to be tokenized
    :return: tokenized list of the 'text', with punctuation and one-letter words removed
    """
    sklearn_tokenizer = feature_extraction.CountVectorizer().build_tokenizer()
    return sklearn_tokenizer(text)


def treebank_detokenize(tokenized_text):
    detokenizer = treebank.TreebankWordDetokenizer()
    return detokenizer.detokenize(tokenized_text)


def porter_stem(tokenized_text):
    """
    Stems the tokenized_text using the Porter Stemmer
    :param tokenized_text: list of tokens
    :return: tokenized_text with each token stemmed (normalised)
    """
    porter_stemmer = nltk.stem.porter.PorterStemmer()
    stemmed_tokenized_text = []

    for token in tokenized_text:
        token_lc = token.lower()
        stemmed_word = porter_stemmer.stem(token_lc)
        stemmed_tokenized_text.append(stemmed_word)

    return stemmed_tokenized_text


def porter_stem_list(text_list):
    """
    Stems the text list using the Porter Stemmer
    :param text_list: List of text to be stemmed
    :return: The 'text_list' with porter stemming applied
    """
    stemmed_list = []

    for text in text_list:
        tokenized_text = sklearn_tokenize(text)
        stemmed_text = porter_stem(tokenized_text)
        detokenized_stemmed_text = treebank_detokenize(stemmed_text)
        stemmed_list.append(detokenized_stemmed_text)

    return stemmed_list


def simple_pos_tag(pos_tag):
    """
    Converts the complex, and informative the Penn Treebank part of speech (POS) tag produce by
    nltk.pos_tag() into the simplified POS tags expected by the WordNetLemmatizer.
    :param pos_tag: a Treebank POS tag
    :return: simplified Treebank POS tag: Noun, Verb, Adjective, Adverb or other (None)
    """
    if pos_tag.startswith('N'):
        simplified_pos_tag = wordnet_corpus.NOUN
    elif pos_tag.startswith('V'):
        simplified_pos_tag = wordnet_corpus.VERB
    elif pos_tag.startswith('J'):
        simplified_pos_tag = wordnet_corpus.ADJ
    elif pos_tag.startswith('R'):
        simplified_pos_tag = wordnet_corpus.ADV
    else:
        simplified_pos_tag = None
    return simplified_pos_tag


def wordnet_lemmatize(tokenized_text):
    """
    Lemmatizes the tokenized_text using the NLTK WordNet lemmatizer
    :param tokenized_text: list of tokens
    :return: tokenized_text with each token lemmatized (normalised)
    """
    lemmatized_tokens = []
    wordnet_lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    for token, pos in nltk.pos_tag(tokenized_text):
        simplified_pos_tag = simple_pos_tag(pos)
        if simplified_pos_tag is not None:
            lemmatized_token = wordnet_lemmatizer.lemmatize(token, simplified_pos_tag)
            lemmatized_tokens.append(lemmatized_token)
        else:
            lemmatized_tokens.append(token)

    return lemmatized_tokens


def wordnet_lemmatize_list(text_list):
    """Lemmatizes the text list using the Wordnet Lemmatizer
    :param text_list: list of text to be lemmatized
    :return: the text_list with wordnet lemmatisation applied
    """
    lemmatized_list = []

    for text in text_list:
        tokenized_text = sklearn_tokenize(text)
        lemmatized_text = wordnet_lemmatize(tokenized_text)
        detokenized_lemmatized_text = treebank_detokenize(lemmatized_text)
        lemmatized_list.append(detokenized_lemmatized_text)

    return lemmatized_list