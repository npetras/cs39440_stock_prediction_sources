"""
Running of machine learning model
"""

from main.data import clean

def with_vectorizer(train_data, train_labels, test_data, test_lables, classifier, stemming,
                    lemmatization, stop_words, adv_stop_words):

    processed_train_data = []
    processed_test_data = []

    if stemming:
        processed_train_data = clean.porter_stem_list(train_data)
        processed_test_data = clean.porter_stem_list(test_data)
    elif lemmatization:
        processed_train_data = clean.wordnet_lemmatize_list(train_data)
        processed_test_data = clean.wordnet_lemmatize_list(test_data)
    else:
        # error
        pass

