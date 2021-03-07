"""
Contains all functions used to clean the data/remove noise from the data.
"""

import sklearn.feature_extraction.text as feature_extraction
import nltk
import nltk.stem.wordnet
import nltk.stem.porter
import nltk.corpus.reader.wordnet as wordnet_corpus


def sklearn_tokenize(text):
    """
    Runs the sklearn tokenizer on the text provided.
    :param text: string of text that needs to be tokenized
    :return: tokenized list of the 'text', with punctuation and one-letter words removed
    """
    sklearn_tokenizer = feature_extraction.CountVectorizer().build_tokenizer()
    return sklearn_tokenizer(text)


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


def simple_pos_tag(pos_tag):
    """
    Provides the simple part of speech (POS) tag that the WordNetLemmatizer expects, converting the more complex Penn
    Treebank POS tags
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


def word_net_lemmatize(tokenized_text):
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
