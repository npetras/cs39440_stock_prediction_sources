import pickle
import constants
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.naive_bayes
from main.data import load
from main.modelling import run

if __name__ == '__main__':
    # open the pickled file
    with open('sentiment_list.pkl', 'rb') as sentiment_file:
        headline_sentiment = pickle.load(sentiment_file)
    print(headline_sentiment)
    # open csv file for labels
    djia_df = load.from_csv(constants.DJIA_DATA_REL_PATH)
    headline_labels = djia_df['Label']
    # vectorize the list of dicts using DictVectorizer
    # Use cross validation and modelling to run on the data
    run.cross_validation(headline_sentiment, headline_labels,
                         sklearn.feature_extraction.DictVectorizer(dtype=float),
                         sklearn.linear_model.LogisticRegression(max_iter=constants.ITERATION_NUM))
    run.cross_validation(headline_sentiment, headline_labels,
                         sklearn.feature_extraction.DictVectorizer(dtype=float),
                         sklearn.naive_bayes.BernoulliNB())
