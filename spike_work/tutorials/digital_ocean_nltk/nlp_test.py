"""
https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
"""
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist

import re
import string

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

# tokenize the data
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

print(tweet_tokens[0])

# data normalisation
# def lemmatize_sentence(tokens):
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_sentence = []

#     for word, tag in pos_tag(tokens):
#         if tag.startswith('NN'):
#             pos = 'n'
#         elif tag.startswith('VB'):
#             pos = 'v'
#         else:
#             pos = 'a'
#         lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    
#     return lemmatized_sentence

# print(lemmatize_sentence(tweet_tokens[0]))

# removing noise from data
def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        # remove hyperlinks
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        # remove ats (@s)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)


        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    
    return cleaned_tokens
 
stop_words = stopwords.words('english')

# print(remove_noise(tweet_tokens[0], stop_words))

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_clean_tokens_list = []
negative_clean_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_clean_tokens_list.append(remove_noise(tokens))
for tokens in positive_tweet_tokens:
    negative_clean_tokens_list.append(remove_noise(tokens))

# print(positive_tweet_tokens[500])
# print(positive_clean_tokens_list[500])

# text analysis
# generator function
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_clean_tokens_list)
freq_dist_pos = FreqDist(all_pos_words)
# print(freq_dist_pos.most_common(10))