import pandas as pd
from nltk.tokenize.regexp import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.classify import scikitlearn
from nltk.corpus import stopwords

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
    cv_wo_preprocess = CountVectorizer()
    cv_train_basic = cv_wo_preprocess.fit_transform(train_headlines_data)
    cv_test_basic = cv_wo_preprocess.transform(test_headlines_data)
    cv_feat_names_basic = cv_wo_preprocess.get_feature_names()
    filtered_list = filter(lambda x: x == 'for', cv_feat_names_basic)
    print("Count Vectorizer Features: ")
    print(f"{cv_feat_names_basic[1000:1200]}\n")
    print(f"feature length: {len(cv_feat_names_basic)}")

    # Naive Bayes
    nb_model = train_eval_model(cv_train_basic, train_df['Label'], MultinomialNB(), cv_test_basic, test_df['Label'])
    nb_word_features = cv_wo_preprocess.get_feature_names()
    print_coefficients(nb_model, nb_word_features)
    # Logisitic Regression
    logr_model = train_eval_model(cv_train_basic, train_df['Label'], LogisticRegression(), cv_test_basic, test_df['Label'])
    logr_word_features = cv_wo_preprocess.get_feature_names()
    print_coefficients(logr_model, logr_word_features)

    # apply stop word removal and check the impact on the results
    nltk_stop_words = stopwords.words('english')
    cv_stopwords = CountVectorizer(stop_words=nltk_stop_words)
    cv_train_stopwords = cv_stopwords.fit_transform(train_headlines_data)
    cv_test_stopwords = cv_stopwords.transform(test_headlines_data)
    stopword_cv_features = cv_stopwords.get_feature_names()

    nb_stopwords_model = train_eval_model(cv_train_stopwords, train_df['Label'], MultinomialNB(), cv_test_stopwords, test_df['Label'])
    print_coefficients(nb_stopwords_model, stopword_cv_features)

    logr_stopwords_model = train_eval_model(cv_train_stopwords, train_df['Label'], LogisticRegression(), cv_test_stopwords, test_df['Label'])
    print_coefficients(logr_stopwords_model, stopword_cv_features)
