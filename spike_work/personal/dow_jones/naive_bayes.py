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

def train_eval_model(train_data, train_labels, classifier, test_data, test_labels):
    model = classifier.fit(train_data, train_labels)
    score = model.score(test_data, test_labels)
    print(f"{classifier.__class__.__name__} Accuracy: {score}\n")
    return model

def print_coefficients(model, word_features):
    coefficients = model.coef_.tolist()[0]
    feature_weighting = pd.DataFrame({'Word' : word_features, 'Coefficient' : coefficients})
    sorted_feature_weighting = feature_weighting.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
    print("Top positive features:")
    print(sorted_feature_weighting.head(10))
    print("Top negative features:")
    print(sorted_feature_weighting.tail(10))

if __name__ == "__main__":
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

    # bag of words, no pre-processing
    # feature extraction
    count_vectorizer = CountVectorizer()
    vectorized_train_data = count_vectorizer.fit_transform(train_headlines_data)
    vectorized_test_data = count_vectorizer.transform(test_headlines_data)
    feature_names = count_vectorizer.get_feature_names()
    filtered_list = filter(lambda x: x == 'for', feature_names)
    print("Count Vectorizer Features: ")
    print(f"{feature_names[1000:1200]}\n")
    print(f"feature length: {len(feature_names)}")
    for item in filtered_list:
        print(item)

    # Naive Bayes
    nb_model = train_eval_model(vectorized_train_data, train_df['Label'], MultinomialNB(), vectorized_test_data, test_df['Label'])
    nb_word_features = count_vectorizer.get_feature_names()
    print_coefficients(nb_model, nb_word_features)
    # Logisitic Regression
    lr_model = train_eval_model(vectorized_train_data, train_df['Label'], LogisticRegression(), vectorized_test_data, test_df['Label'])
    lr_word_features = count_vectorizer.get_feature_names()
    print_coefficients(lr_model, lr_word_features)

