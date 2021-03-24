"""
Used to extract sentiment from headlines
"""

import logging
import nltk.sentiment.vader

MAX_TWEET_LENGTH = 280


def compound_score(text):
    """
    Provides the compound sentiment score of the text provided, using the VADER sentiment
    analyser. Expects text to be less than 280 characters
    :param text: text to be analysed for sentiment
    :return: Compound sentiment score
    """
    # warning, due to vader being trained on short text -- social media posts
    if len(text) > MAX_TWEET_LENGTH:
        logging.warning('Text provided is longer than a standard tweet, VADER is designed to be '
                        'used on short pieces of text')
    vader = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    return vader.polarity_scores(text)['compound']
