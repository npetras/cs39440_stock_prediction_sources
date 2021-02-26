import pandas as pd
from nltk.tokenize.regexp import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.classify import scikitlearn

from preprocessing import preprocess
import constants

def combine_headline_rows(data_frame):
    combined_headlines = []

    for row in range(0, len(data_frame.index)):
        combined_headlines_in_row = ''
        for headline in data_frame.iloc[row, 2:27]:
            headline_str = str(headline)
            combined_headlines_in_row = combined_headlines_in_row + ' ' + headline_str
        # remove dangling spaces at beginning and end, trim?
        combined_headlines.append(combined_headlines_in_row.strip())

    return combined_headlines

# import and split data
djia_df= pd.read_csv(constants.ABS_PATH_TO_DATASET)
train_df = djia_df[djia_df['Date'] < '2015-01-01']
test_df = djia_df[djia_df['Date'] > '2014-12-31']

# basic preprocessing that is required
# combine each row into one string
train_headlines_data = combine_headline_rows(train_df)
print("Train Headlines:")
print(train_headlines_data[0] + '\n')

test_headlines_data = combine_headline_rows(test_df)
print("Test Headlines:")
print(test_headlines_data[0] + '\n')

# count vectorizer, sklearn approach
# feature extraction
count_vectorizer = CountVectorizer()
vectorized_train_data = count_vectorizer.fit_transform(train_headlines_data)
vectorized_test_data = count_vectorizer.transform(test_headlines_data)

# training model
naive_bayes_classifier = MultinomialNB()
naive_bayes_model = naive_bayes_classifier.fit(vectorized_train_data, train_df['Label'])
# test/evaluation
nb_prediction_score = naive_bayes_model.score(vectorized_test_data, test_df['Label'])
print(f"Sklearn MultinomiaNB Accuracy: {nb_prediction_score}\n")

nb_word_features = count_vectorizer.get_feature_names()
nb_coefficients = naive_bayes_model.coef_.tolist()[0]
nb_feature_weighting = pd.DataFrame({'Word' : nb_word_features, 'Coefficient' : nb_coefficients})
nb_sorted_feature_weighting = nb_feature_weighting.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
print(nb_sorted_feature_weighting.head(10))
print(nb_sorted_feature_weighting.tail(10))

log_reg_classifier = LogisticRegression()
log_reg_model = log_reg_classifier.fit(vectorized_train_data, train_df['Label'])
lr_prediction_score = log_reg_model.score(vectorized_test_data, test_df['Label'])
print(f"Sklearn LogisticRegression Accuracy: {lr_prediction_score}\n")

lr_word_features = count_vectorizer.get_feature_names()
lr_coefficients = log_reg_model.coef_.tolist()[0]
lr_feature_weighting = pd.DataFrame({'Word' : lr_word_features, 'Coefficient' : lr_coefficients})
lr_sorted_feature_weighting = lr_feature_weighting.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
print(lr_sorted_feature_weighting.head(10))
print(lr_sorted_feature_weighting.tail(10))

