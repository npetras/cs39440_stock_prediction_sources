import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.tsv', sep='\t')
print(data.head()) 

sentiment_count = data.groupby('Sentiment').count()
plt.bar(sentiment_count.index.values, sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Reviews')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts = cv.fit_transform(data['Phrase'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'], test_size = 0.3, random_state = 1)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Multinomial NB Accuracy: ", metrics.accuracy_score(y_test, predicted))
