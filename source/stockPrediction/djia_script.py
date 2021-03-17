"""
Script that runs models on the DJIA dataset
"""

from main.data import load, manipulate
from main.classifier import run

from sklearn import linear_model, naive_bayes

DJIA_DATA_REL_PATH = '../../datasets/existing/dow_jones/Combined_News_DJIA.csv'
ITERATION_NUM = 500


def combine_djia_headlines(data_frame):
    combined_headlines = []
    for row in range(0, len(data_frame.index)):
        combined_headlines_in_row = manipulate.combine_strings(
            data_frame.iloc[row, 2:27])
        combined_headlines.append(combined_headlines_in_row)
    return combined_headlines


if __name__ == '__main__':
    djia_df = load.from_csv(DJIA_DATA_REL_PATH)
    train_df = djia_df[djia_df['Date'] < '2015-01-01']
    test_df = djia_df[djia_df['Date'] > '2014-12-31']

    data = combine_djia_headlines(djia_df)
    train_data = combine_djia_headlines(train_df)
    test_data = combine_djia_headlines(test_df)

    print("Multinomial Naive Bayes -- Without Extra Preprocessing, Holdout")
    run.with_vectorizer(train_data=train_data,
                        train_labels=train_df['Label'],
                        test_data=test_data,
                        classifier=naive_bayes.MultinomialNB(),
                        test_labels=test_df['Label'])

    # print("Logistic Regression -- Without Extra Preprocessing")
    # run.with_vectorizer(train_data=train_data,
    #                     train_labels=train_df['Label'],
    #                     test_data=test_data,
    #                     classifier=linear_model.LogisticRegression(max_iter=ITERATION_NUM),
    #                     test_labels=test_df['Label'])
    #
    # print("Multinomial Naive Bayes -- Stemming Only")
    # run.with_vectorizer(train_data=train_data,
    #                     train_labels=train_df['Label'],
    #                     test_data=test_data,
    #                     classifier=naive_bayes.MultinomialNB(),
    #                     test_labels=test_df['Label'],
    #                     stemming=True)
    #
    # print("Logistic Regression -- Stemming, Stopword and Frequency Removal")
    # run.with_vectorizer(
    #     train_data=train_data,
    #     train_labels=train_df['Label'],
    #     test_data=test_data,
    #     classifier=linear_model.LogisticRegression(max_iter=ITERATION_NUM),
    #     test_labels=test_df['Label'],
    #     stop_words=True,
    #     stemming=True,
    #     frequency_removal=True)

    print("Multinomial Naive Bayes -- Without Extra Preprocessing, Cross Validation")
    run.with_vectorizer_cv(data, djia_df['Label'], classifier=naive_bayes.MultinomialNB())
