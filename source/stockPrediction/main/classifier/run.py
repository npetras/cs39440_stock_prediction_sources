"""
Running of machine learning classifier
"""

from nltk.corpus import stopwords
import sklearn.feature_extraction.text as feature_extraction
from sklearn import naive_bayes
import pandas as pd

from main.data import clean

MAX_FREQ_CAP = 0.97
MIN_FREQ_CAP = 0.03
STOPWORDS_LIST = stopwords.words('english')


# pylint: disable=R0913,R0914
# Arguments, and variables should be reduced in future
def with_vectorizer(train_data, train_labels, test_data, test_labels,
                    classifier=naive_bayes.MultinomialNB(), stemming=False,
                    lemmatization=False, stop_words=False, frequency_removal=False):
    """
    Runs a classifier on training and test data provided, using a CountVectorizer (bag of words)
    approach.
    Stop words are removed based on a list.
    Frequency removal removes words that occur in more than 97% of the documents, and words that
    occur in less than 3% of the documents.

    :param train_data: Training data, without labels - list of strings
    :param train_labels: Labels that are associated with the train_data - list of labels (strings,
    numbers, floats or similar)
    :param test_data: Test data, without labels - list of strings
    :param test_labels: Labels that are associated with the test_data - list of labels (strings,
    numbers, floats or similar)
    :param classifier: The classifier that should be used on the data - currently only expect
    MultinomialNB and LogisticRegression
    :param stemming: Should stemming be used or not,  mutually exclusive with lemmatization -
    expect True/False
    :param lemmatization: Should lemmatisation be used or not, mutually exclusive with stemming
     - expect True/False
    :param stop_words: Should stopwords be removed based on a list - expect True/False
    :param frequency_removal: Should the words that occur in more than 97% of documents, or in less
    than 3% of documents - expect True/False

    :return: None
    """
    processed_train_data = train_data
    processed_test_data = test_data

    if stemming and lemmatization:
        raise Exception('Cannot use both stemming and lemmatization')

    if stemming:
        processed_train_data = clean.porter_stem_list(train_data)
        processed_test_data = clean.porter_stem_list(test_data)
    elif lemmatization:
        processed_train_data = clean.wordnet_lemmatize_list(train_data)
        processed_test_data = clean.wordnet_lemmatize_list(test_data)

    count_vectorizer = create_count_vectorizer(frequency_removal, stop_words)

    vectorized_train_data = count_vectorizer.fit_transform(processed_train_data)
    vectorizer_test_data = count_vectorizer.transform(processed_test_data)

    model = classifier.fit(vectorized_train_data, train_labels)
    score = model.score(vectorizer_test_data, test_labels)
    print(f"{classifier.__class__.__name__} Accuracy: {score}\n")

    # coefficients, top features
    word_features = count_vectorizer.get_feature_names()
    print_coefficients(model, word_features)


def create_count_vectorizer(frequency_removal, stop_words):
    """
    Create a CountVectorizer based on the paramaters provided.
    :param frequency_removal: Should the words that occur in more than 97% of documents, or in less
    than 3% of documents - expect True/False
    :param stop_words: Should stopwords be removed based on a list - expect True/False
    :return: The specified CountVectorizer object
    """
    if stop_words and frequency_removal:
        count_vectorizer = feature_extraction.CountVectorizer(
            stop_words=stopwords.words('english'),
            max_df=MAX_FREQ_CAP,
            min_df=MIN_FREQ_CAP
        )
    elif stop_words:
        count_vectorizer = feature_extraction.CountVectorizer(stop_words=STOPWORDS_LIST)
    elif frequency_removal:
        count_vectorizer = feature_extraction.CountVectorizer(max_df=MAX_FREQ_CAP,
                                                              min_df=MIN_FREQ_CAP)
    else:
        count_vectorizer = feature_extraction.CountVectorizer()
    return count_vectorizer


def print_coefficients(model, word_features):
    """
    Prints the top features (co-efficients) of the classifier provided
    :param model: The classifier for which the coefficients are required
    :param word_features: All of the word features for the 'classifier'
    :return: None
    """
    coefficients = model.coef_.tolist()[0]
    feature_weighting = pd.DataFrame({'Word': word_features, 'Coefficient': coefficients})
    sorted_feature_weighting = feature_weighting.sort_values(['Coefficient', 'Word'],
                                                             ascending=[0, 1])
    print("Top 10 positive features:")
    print(sorted_feature_weighting.head(10))
    print("Top 10 negative features:")
    print(sorted_feature_weighting.tail(10))
    print()
