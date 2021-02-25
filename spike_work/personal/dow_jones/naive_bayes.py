import pandas as pd
from nltk.tokenize.regexp import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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


def get_data_for_model(tokens_list):
    for tokens in tokens_list:
        yield dict([token, True] for token in tokens)


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
print("Sklearn Vectorizer Shape: ")
print(f'{vectorized_train_data.shape}\n')

# training model
naive_bayes_classifier = MultinomialNB()
naive_bayes_model = naive_bayes_classifier.fit(vectorized_train_data, train_df['Label'])

# test/evaluation
vectorized_test_data = count_vectorizer.transform(test_headlines_data)
prediction_score = naive_bayes_model.score(vectorized_test_data, test_df['Label'])
print(f"Sklearn MultinomiaNB Accuracy: {prediction_score}\n")

# nltk manual approach
tokenized_train_headlines = []

for headlines in train_headlines_data:
    tokenized_headlines = WordPunctTokenizer().tokenize(headlines)
    tokenized_train_headlines.append(tokenized_headlines)

print("NLTK: Tokenized Train Data: ")
print(f'{tokenized_train_headlines[0]}\n')

train_data_dict_list = get_data_for_model(tokenized_train_headlines)

positive_dataset = [(train_data_dict_list, train_df['Label'])
                         for train_data_dict in tokenized_train_headlines]

train_data_for_model = []

for index in enumerate(train_data_dict_list):
    dict_label_tuple = (train_data_dict_list[index], train_df['Label'][index]
    train_data_for_model.append(dict_label_tuple)
