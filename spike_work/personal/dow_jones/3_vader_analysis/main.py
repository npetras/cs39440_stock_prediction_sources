import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import constants
import clean_data

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

def preprocess_data(headlines_list):
    processed_headlines_list = []

    for headlines in headlines_list:
        tokenized_headlines = clean_data.sklearn_tokenize(headlines)
        stemmed_headlines = clean_data.porter_stem(tokenized_headlines)
        detokenized_lemmatized_headlines = clean_data.detokenize(stemmed_headlines)
        processed_headlines_list.append(detokenized_lemmatized_headlines)
    
    return processed_headlines_list
        
def create_vader_sent_labels(headlines_list):
    vader_sentiment_labels = []
    for headlines in headlines_list:
        vader_score = vader.polarity_scores(headlines)
        vader_com_score = vader_score['compound']
        vader_sentiment_labels.append(vader_com_score)
    return vader_sentiment_labels

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
    print()

if __name__ == "__main__":
    # import and split data
    djia_df= pd.read_csv(constants.ABS_PATH_TO_DATASET)
    train_df = djia_df[djia_df['Date'] < '2015-01-01']
    test_df = djia_df[djia_df['Date'] > '2014-12-31']

    # basic preprocessing that is required
    # combine each row into one string
    train_headlines_data = combine_headline_rows(train_df)
    test_headlines_data = combine_headline_rows(test_df)
    
    # preprocess text
    processed_train_data = preprocess_data(train_headlines_data)
    processed_test_data = preprocess_data(test_headlines_data)

    # convert to vader compound score
    vader = SentimentIntensityAnalyzer()
    vader_train_sent_labels = create_vader_sent_labels(processed_train_data)
    vader_test_sent_labels = create_vader_sent_labels(processed_test_data)


    cv = CountVectorizer(lowercase=False)
    cv_train = cv.fit_transform(vader_train_sent_labels)
    cv_test = cv.transform(vader_test_sent_labels)
    cv_features = cv.get_feature_names()



    # run the model on the VADER Sentiment Scores
