"""
Runs model on DJIA dataset after sentiment has been extracted from the data.
"""

from main.feature.engineering import sentiment
from main.data import clean
from main.data import load
import constants
import logging


def preprocess_headline(headline):
    preprocessed_headline = clean.sklearn_tokenize(headline)
    preprocessed_headline = clean.wordnet_lemmatize_list(preprocessed_headline)
    preprocessed_headline = clean.treebank_detokenize(preprocessed_headline)
    preprocessed_headline = preprocessed_headline.lower()
    preprocessed_headline = clean.sklearn_tokenize(preprocessed_headline)
    preprocessed_headline = clean.remove_stopwords(preprocessed_headline)
    preprocessed_headline = clean.treebank_detokenize(preprocessed_headline)
    return preprocessed_headline


def extract_sentiment_for_each_row(data_frame):
    headline_combined_sentiment = []
    for row in range(0, len(data_frame.index)):
        row_polarity_score = 0.00
        for col in range(2, len(data_frame.columns)):
            headline = str(data_frame.iloc[row, col])
            processed_headline = preprocess_headline(headline)
            polarity_score = sentiment.compound_score(processed_headline)
            row_polarity_score += polarity_score
            print(f'Headline: {processed_headline}')
            print(f'Compound score {polarity_score}')

        if row_polarity_score > 0.00:
            row_sentiment = {'sentiment': 'positive'}
        else:
            row_sentiment = {'sentiment': 'negative'}
        print(f'Row compound score: {row_polarity_score}')
        print(f'Sentiment: {row_sentiment}\n')
        headline_combined_sentiment.append(row_sentiment)
    return headline_combined_sentiment


if __name__ == '__main__':
    djia_df = load.from_csv(constants.DJIA_DATA_REL_PATH)

    headline_sentiment = extract_sentiment_for_each_row(djia_df.tail(50))
    print(headline_sentiment)


