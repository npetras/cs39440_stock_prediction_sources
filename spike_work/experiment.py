# source: 
# https://towardsdatascience.com/cleaning-preprocessing-text-data-for-sentiment-analysis-382a41f150d6
import re
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import spacy

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def space(comment):
    doc = nlp(comment)
    return " ".join([token.lemma_ for token in doc])

# read and load in the file
df = pd.read_csv('amazon_alexa.tsv', sep = '\t')
print(df.head())
nlp = spacy.load('en', disable=['parser', 'ner'])

# creates new column 'new_reviews' replacing the 'verified_reviews' column
# converting verified_reviews column to lowercase
df['new_reviews'] = df['verified_reviews'].apply(lambda x: " ".join(x.lower() for x in x.split()))
print(df['new_reviews'].head())
print(df.head())

# remove punctuation 
df['new_reviews'] = df['new_reviews'].str.replace('[^\w\s]', '')
print(df['new_reviews'].head())

# remove emojis
df['new_reviews'] = df['new_reviews'].apply(lambda x: remove_emoji(x))

# remove stopwords
stop = stopwords.words('english')
df['new_reviews'] = df['new_reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(df.head(20))

# lemmataization
df['new_reviews'] = df['new_reviews'].apply(space)
print(df.head(20))