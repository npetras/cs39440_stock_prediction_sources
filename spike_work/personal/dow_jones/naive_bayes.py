import pandas as pd
import nltk.classify.scikitlearn

from preprocessing import preprocess
import constants

def get_data_for_model(tokens_list):
    for tokens in tokens_list:
        yield dict([token, True] for token in tokens)


# import and split data
data_frame = pd.read_csv(constants.PATH_TO_DATASET)
train_df = data_frame[data_frame['Date'] < '2015-01-01']
test_df = data_frame[data_frame['Date'] > '2014-12-31']

# basic preprocessing that is required

# combine each row into one string
train_headlines_data = []

for row in range(0, len(train_df.index)):
    combined_headlines_in_row = ''
    for headline in train_df.iloc[row, 2:27]:
        headline_str = str(headline)
        # removes bs and start and end quotes from headlines
        headline_str = preprocess.remove_b(headline_str)
        headline_str = preprocess.remove_start_end_quotes(headline_str)
        combined_headlines_in_row = combined_headlines_in_row + ' ' + headline_str

    # remove dangling spaces at beginning and end, trim?
    train_headlines_data.append(combined_headlines_in_row.strip())

print(train_headlines_data[0])

# train model on data with no preprocessing

