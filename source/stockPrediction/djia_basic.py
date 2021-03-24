"""
Script that runs models on the DJIA dataset.
Uses bag of words vector as features. No sentiment analysis, only uses the words to perform price
prediction.
"""

from main.data import load, manipulate
from main.modelling import run
import constants

from sklearn import linear_model, naive_bayes



def combine_djia_headlines(data_frame):
    combined_headlines = []
    for row in range(0, len(data_frame.index)):
        combined_headlines_in_row = manipulate.combine_strings(
            data_frame.iloc[row, 2:27])
        combined_headlines.append(combined_headlines_in_row)
    return combined_headlines


if __name__ == '__main__':
    djia_df = load.from_csv(constants.DJIA_DATA_REL_PATH)
    train_df = djia_df[djia_df['Date'] < '2015-01-01']
    test_df = djia_df[djia_df['Date'] > '2014-12-31']

    data = combine_djia_headlines(djia_df)
    train_data = combine_djia_headlines(train_df)
    test_data = combine_djia_headlines(test_df)

    # print("Multinomial Naive Bayes -- Without Extra Preprocessing, Holdout")
    # run.with_vectorizer(train_data=train_data,
    #                     train_labels=train_df['Label'],
    #                     test_data=test_data,
    #                     modelling=naive_bayes.MultinomialNB(),
    #                     test_labels=test_df['Label'])
    #
    # print("Logistic Regression -- Without Extra Preprocessing")
    # run.with_vectorizer(
    #     train_data=train_data,
    #     train_labels=train_df['Label'],
    #     test_data=test_data,
    #     modelling=linear_model.LogisticRegression(max_iter=ITERATION_NUM),
    #     test_labels=test_df['Label'])

    # print("Multinomial Naive Bayes -- Stemming Only")
    # run.with_vectorizer(train_data=train_data,
    #                     train_labels=train_df['Label'],
    #                     test_data=test_data,
    #                     modelling=naive_bayes.MultinomialNB(),
    #                     test_labels=test_df['Label'],
    #                     stemming=True)
    #
    # print("Logistic Regression -- Stemming, Stopword and Frequency Removal")
    # run.with_vectorizer(
    #     train_data=train_data,
    #     train_labels=train_df['Label'],
    #     test_data=test_data,
    #     modelling=linear_model.LogisticRegression(max_iter=ITERATION_NUM),
    #     test_labels=test_df['Label'],
    #     stop_words=True,
    #     stemming=True,
    #     frequency_removal=True)

    print("MultinomialNB -- CV - Count Vectorizer")
    run.with_vectorizer_cv(data,
                           djia_df['Label'],
                           classifier=naive_bayes.MultinomialNB())
    print("MultinomialNB -- CV - Stopwords")
    run.with_vectorizer_cv(data,
                           djia_df['Label'],
                           classifier=naive_bayes.MultinomialNB(),
                           stop_words=True)
    print("MultinomialNB -- CV - Frequency Removal")
    run.with_vectorizer_cv(data,
                           djia_df['Label'],
                           classifier=naive_bayes.MultinomialNB(),
                           frequency_removal=True)
    print("MultinomialNB -- CV - Stemming")
    run.with_vectorizer_cv(data=data,
                           labels=djia_df['Label'],
                           classifier=naive_bayes.MultinomialNB(),
                           stemming=True)
    print("MultinomialNB -- CV - Stemming & Stopwords")
    run.with_vectorizer_cv(data=data,
                           labels=djia_df['Label'],
                           classifier=naive_bayes.MultinomialNB(),
                           stemming=True,
                           stop_words=True)
    print("MultinomialNB -- CV -- Stemming, Stopword & Frequency Removal")
    run.with_vectorizer_cv(data=data,
                           labels=djia_df['Label'],
                           classifier=naive_bayes.MultinomialNB(),
                           stop_words=True,
                           stemming=True,
                           frequency_removal=True)
    print("MultinomialNB -- CV -- Lemmatisation")
    run.with_vectorizer_cv(data=data,
                           labels=djia_df['Label'],
                           classifier=naive_bayes.MultinomialNB(),
                           lemmatization=True)
    print("MultinomialNB -- CV -- Lemmatisation & Stopword")
    run.with_vectorizer_cv(data=data,
                           labels=djia_df['Label'],
                           classifier=naive_bayes.MultinomialNB(),
                           lemmatization=True,
                           stop_words=True)
    print("MultinomialNB -- CV -- Lemmatisation, Stopword & Frequency Removal")
    run.with_vectorizer_cv(data=data,
                           labels=djia_df['Label'],
                           classifier=naive_bayes.MultinomialNB(),
                           stop_words=True,
                           lemmatization=True,
                           frequency_removal=True)

    print("LogisticRegression -- CV -- Count Vectorizer")
    run.with_vectorizer_cv(
        data,
        djia_df['Label'],
        classifier=linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM))

    print("LogisticRegression -- CV -- Stopwords")
    run.with_vectorizer_cv(
        data,
        djia_df['Label'],
        classifier=linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM),
        stop_words=True)

    print("LogisticRegression -- CV -- Frequency Removal")
    run.with_vectorizer_cv(
        data,
        djia_df['Label'],
        classifier=linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM),
        frequency_removal=True)

    print("LogisticRegression -- CV -- Stemming")
    run.with_vectorizer_cv(
        data=data,
        labels=djia_df['Label'],
        classifier=linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM),
        stemming=True)

    print("LogisticRegression -- CV -- Stemming & Stopwords")
    run.with_vectorizer_cv(
        data=data,
        labels=djia_df['Label'],
        classifier=linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM),
        stemming=True,
        stop_words=True)

    print("LogisticRegression -- CV -- Stemming, Stopword & Frequency Removal")
    run.with_vectorizer_cv(
        data=data,
        labels=djia_df['Label'],
        classifier=linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM),
        stop_words=True,
        stemming=True,
        frequency_removal=True)

    print("LogisticRegression -- Lemmatisation")
    run.with_vectorizer_cv(
        data=data,
        labels=djia_df['Label'],
        classifier=linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM),
        lemmatization=True)
    print("LogisticRegression -- CV -- Lemmatisation & Stopword")
    run.with_vectorizer_cv(
        data=data,
        labels=djia_df['Label'],
        classifier=linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM),
        stop_words=True,
        lemmatization=True)
    print(
        "LogisticRegression -- CV -- Lemmatisation, Stopword & Frequency Removal"
    )
    run.with_vectorizer_cv(
        data=data,
        labels=djia_df['Label'],
        classifier=linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM),
        stop_words=True,
        lemmatization=True,
        frequency_removal=True)
