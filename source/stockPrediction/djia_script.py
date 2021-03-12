"""
Script that runs models on the DJIA dataset
"""

from main.data import load, manipulate
from main.model import run

from sklearn import linear_model, naive_bayes

DJIA_DATA_REL_PATH = '../../datasets/existing/dow_jones/Combined_News_DJIA.csv'


def combine_djia_headlines(data_frame):
    combined_headlines = []
    for row in range(0, len(data_frame.index)):
        combined_headlines_in_row = manipulate.combine_strings(data_frame.iloc[row, 2:27])
        combined_headlines.append(combined_headlines_in_row)
    return combined_headlines


if __name__ == '__main__':
    djia_df = load.from_csv(DJIA_DATA_REL_PATH)
    train_df = djia_df[djia_df['Date'] < '2015-01-01']
    test_df = djia_df[djia_df['Date'] > '2014-12-31']

    train_data = combine_djia_headlines(train_df)
    test_data = combine_djia_headlines(test_df)

    run.with_vectorizer(train_data=train_data, train_labels=train_df['Label'], test_data=test_data,
                        classifier=naive_bayes.MultinomialNB(), test_labels=test_df['Label'],
                        stemming=True)
    run.with_vectorizer(train_data=train_data, train_labels=train_df['Label'], test_data=test_data,
                        classifier=linear_model.LogisticRegression(max_iter=250),
                        test_labels=test_df['Label'], stemming=True, frequency_removal=True)
