"""
Running of machine learning modelling.
Methods are split by the type of Text Representation being used and the train/test split being
used.
"""

from nltk.corpus import stopwords
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.pipeline
import sklearn.model_selection
import pandas as pd

from main.data import clean

MAX_FREQ_CAP = 0.97
MIN_FREQ_CAP = 0.03
STOPWORDS_LIST = stopwords.words('english')


# pylint: disable=R0913,R0914
# Arguments, and variables should be reduced in future
def with_vectorizer(train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    classifier=sklearn.naive_bayes.MultinomialNB(),
                    stemming=False,
                    lemmatization=False,
                    stop_words=False,
                    frequency_removal=False):
    """
    Runs a modelling on training and test data provided, using a CountVectorizer (bag of words)
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
    :param classifier: The modelling that should be used on the data - currently only expect
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
    feature_probabities = None
    if isinstance(classifier, sklearn.naive_bayes.MultinomialNB):
        feature_probabities = model.feature_log_prob_.tolist()[1]
    elif isinstance(classifier, sklearn.linear_model.LogisticRegression):
        feature_probabities = model.coef_.tolist()[0]

    word_features = count_vectorizer.get_feature_names()
    print_top_features(feature_probabities, word_features)


def create_count_vectorizer(frequency_removal, stop_words):
    """
    Create a CountVectorizer based on the paramaters provided.
    :param frequency_removal: Should the words that occur in more than 97% of documents, or in less
    than 3% of documents - expect True/False
    :param stop_words: Should stopwords be removed based on a list - expect True/False
    :return: The specified CountVectorizer object
    """
    if stop_words and frequency_removal:
        count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            stop_words=stopwords.words('english'),
            max_df=MAX_FREQ_CAP,
            min_df=MIN_FREQ_CAP)
    elif stop_words:
        count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            stop_words=STOPWORDS_LIST)
    elif frequency_removal:
        count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            max_df=MAX_FREQ_CAP, min_df=MIN_FREQ_CAP)
    else:
        count_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    return count_vectorizer


def print_top_features(feature_weightings, word_features):
    """
    Prints the top positive and negative feature for the modelling. Only for binary classification.
    :param feature_weightings: How predictive each feature is for the positive label
    :param word_features: All of the word features for the 'modelling'
    :return: None
    """
    feature_weighting = pd.DataFrame({
        'Word': word_features,
        'Feature Weighting': feature_weightings
    })
    sorted_feature_weighting = feature_weighting.sort_values(
        ['Feature Weighting', 'Word'], ascending=[0, 1])
    print("Top 10 positive features:")
    print(sorted_feature_weighting.head(10))
    print("Top 10 negative features:")
    print(sorted_feature_weighting.tail(10))
    print()


def with_vectorizer_cv(data,
                       labels,
                       classifier=sklearn.naive_bayes.MultinomialNB(),
                       stemming=False,
                       lemmatization=False,
                       stop_words=False,
                       frequency_removal=False):
    """
    Runs the modelling provided on the data provided, with 5 KFold Cross Validation, printing
    the modelling's accuracy and F1 score for both the training and test data - mean scores and
    standard deviation for each.

    :param data: data to be used for as train and test data in cross validation
    :param labels: the labels for the data provided
    :param classifier: the modelling to be used, expects: MultinomialNB and LogisticRegression only
    currently
    :param stemming: Should stemming be used or not,  mutually exclusive with lemmatization -
    expect True/False
    :param lemmatization: Should lemmatisation be used or not, mutually exclusive with stemming
     - expect True/False
    :param stop_words: Should stopwords be removed based on a list - expect True/False
    :param frequency_removal: Should the words that occur in more than 97% of documents, or in less
    than 3% of documents - expect True/False
    """
    processed_data = data

    if stemming and lemmatization:
        raise Exception('Cannot use both stemming and lemmatization')

    if stemming:
        processed_data = clean.porter_stem_list(data)
    elif lemmatization:
        processed_data = clean.wordnet_lemmatize_list(data)

    count_vectorizer = create_count_vectorizer(frequency_removal, stop_words)
    pipeline = sklearn.pipeline.Pipeline([('transformer', count_vectorizer),
                                          ('estimator', classifier)])
    # scores = sklearn.model_selection.cross_val_score(pipeline, processed_data, labels, cv=5)
    scoring_metrics = ['accuracy', 'f1']
    model_output = sklearn.model_selection.cross_validate(
        estimator=pipeline,
        X=processed_data,
        y=labels,
        return_train_score=True,
        scoring=scoring_metrics)
    print(f"{classifier.__class__.__name__} CV Test Accuracy Mean: "
          f"{model_output['test_accuracy'].mean():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Test Accuracy Deviation: "
          f"{model_output['test_accuracy'].std():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Test F1 Mean: "
          f"{model_output['test_f1'].mean():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Test F1 Deviation: "
          f"{model_output['test_f1'].std():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Train Accuracy Mean: "
          f"{model_output['train_accuracy'].mean():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Train Accuracy Deviation: "
          f"{model_output['train_accuracy'].std():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Train F1 Mean: "
          f"{model_output['train_f1'].mean():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Train F1 Deviation: "
          f"{model_output['train_f1'].std():{6}.{4}}\n")


def cross_validation(data, labels, vectorizer, classifier):
    scoring_metrics = ['accuracy', 'f1']
    pipeline = sklearn.pipeline.Pipeline([('transformer', vectorizer),
                                          ('estimator', classifier)])
    model_output = sklearn.model_selection.cross_validate(
        estimator=pipeline,
        X=data,
        y=labels,
        return_train_score=True,
        scoring=scoring_metrics)
    print(f"{classifier.__class__.__name__} CV Test Accuracy Mean: "
          f"{model_output['test_accuracy'].mean():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Test Accuracy Deviation: "
          f"{model_output['test_accuracy'].std():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Test F1 Mean: "
          f"{model_output['test_f1'].mean():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Test F1 Deviation: "
          f"{model_output['test_f1'].std():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Train Accuracy Mean: "
          f"{model_output['train_accuracy'].mean():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Train Accuracy Deviation: "
          f"{model_output['train_accuracy'].std():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Train F1 Mean: "
          f"{model_output['train_f1'].mean():{6}.{4}}")
    print(f"{classifier.__class__.__name__} CV Train F1 Deviation: "
          f"{model_output['train_f1'].std():{6}.{4}}\n")