import pandas as pd
from nltk.corpus import stopwords
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

def stem_all_headlines(headlines_list):
    stemmed_headlines_list = []
    
    for headlines in headlines_list:
        tokenized_headlines = clean_data.sklearn_tokenize(headlines)
        stemmed_headlines = clean_data.porter_stem(tokenized_headlines)
        detokenized_stemmed_headlines = clean_data.detokenize(stemmed_headlines)
        stemmed_headlines_list.append(detokenized_stemmed_headlines)
    
    return stemmed_headlines_list

def lemmatize_all_headlines(headlines_list):
    lemmatized_headlines_list = []

    for headlines in headlines_list:
        tokenized_headlines = clean_data.sklearn_tokenize(headlines)
        lemmatized_headlines = clean_data.word_net_lemmatize(tokenized_headlines)
        detokenized_lemmatized_headlines = clean_data.detokenize(lemmatized_headlines)
        lemmatized_headlines_list.append(detokenized_lemmatized_headlines)
    
    return lemmatized_headlines_list
        

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

    nltk_stop_words = stopwords.words('english')

    # advanced stop words
    print("--- Advanced Stopword Removal ---")

    cv_adv_stopwords = CountVectorizer(stop_words=nltk_stop_words, max_df=0.97, min_df=0.03)
    cv_train_adv_stopwords = cv_adv_stopwords.fit_transform(train_headlines_data)
    cv_test_adv_stopwords = cv_adv_stopwords.transform(test_headlines_data)
    cv_adv_stopwords_features = cv_adv_stopwords.get_feature_names()

    nb_adv_stopwords_model = train_eval_model(cv_train_adv_stopwords, train_df['Label'], MultinomialNB(), cv_test_adv_stopwords, test_df['Label'])
    print_coefficients(nb_adv_stopwords_model, cv_adv_stopwords_features)
    lr_adv_stopwords_model = train_eval_model(cv_train_adv_stopwords, train_df['Label'], LogisticRegression(), cv_test_adv_stopwords, test_df['Label'])
    print_coefficients(lr_adv_stopwords_model, cv_adv_stopwords_features)

    # prediction with stemmed headlines
    # stem the headlines
    stemmed_train_data = stem_all_headlines(train_headlines_data)
    stemmed_test_data = stem_all_headlines(test_headlines_data)
    # print(stemmed_train_data[0])
    # print(stemmed_train_data[1])
    # print()

    # stem only
    print("--- Stemming Only ---")
    cv_stem = CountVectorizer()
    cv_train_stem = cv_stem.fit_transform(stemmed_train_data)
    cv_test_stem = cv_stem.transform(stemmed_test_data)
    cv_stem_features = cv_stem.get_feature_names()
    # print(cv_train_stem.shape)
    # print(cv_test_stem.shape)

    nb_stem_model = train_eval_model(cv_train_stem, train_df['Label'], MultinomialNB(), cv_test_stem, test_df['Label'])
    print_coefficients(nb_stem_model, cv_stem_features)
    lr_stem_model = train_eval_model(cv_train_stem, train_df['Label'], LogisticRegression(), cv_test_stem, test_df['Label'])
    print_coefficients(lr_stem_model, cv_stem_features)


    # stem and stopwords
    print("--- Stemming and Stop Word Removal ----")
    cv_stem_stopwords = CountVectorizer(stop_words=nltk_stop_words)
    cv_train_stem_stopwords = cv_stem_stopwords.fit_transform(stemmed_train_data)
    cv_test_stem_stopwords = cv_stem_stopwords.transform(stemmed_test_data)
    cv_stem_stopwords_features = cv_stem_stopwords.get_feature_names()

    nb_stem_stopwords_model = train_eval_model(cv_train_stem_stopwords, train_df['Label'], MultinomialNB(), cv_test_stem_stopwords, test_df['Label'])
    print_coefficients(nb_stem_stopwords_model, cv_stem_stopwords_features)
    lr_stem_stopwords_model = train_eval_model(cv_train_stem_stopwords, train_df['Label'], LogisticRegression(), cv_test_stem_stopwords, test_df['Label'])
    print_coefficients(lr_stem_stopwords_model, cv_stem_stopwords_features)

    # stem and advanced stop words
    print("--- Stemming and Advanced Stop Word Removal ----")
    cv_stem_stopwords_adv = CountVectorizer(stop_words=nltk_stop_words, max_df=0.97, min_df=0.03)
    cv_train_stem_stopwords_adv = cv_stem_stopwords_adv.fit_transform(stemmed_train_data)
    cv_test_stem_stopwords_adv = cv_stem_stopwords_adv.transform(stemmed_test_data)
    cv_stem_stopwords_adv_feats = cv_stem_stopwords_adv.get_feature_names()

    nb_stem_stopwords_adv_model = train_eval_model(cv_train_stem_stopwords_adv, train_df['Label'], MultinomialNB(), cv_test_stem_stopwords_adv, test_df['Label'])
    print_coefficients(nb_stem_stopwords_adv_model, cv_stem_stopwords_adv_feats)
    lr_stem_stopwords_adv_model = train_eval_model(cv_train_stem_stopwords_adv, train_df['Label'], LogisticRegression(), cv_test_stem_stopwords_adv, test_df['Label'])
    print_coefficients(lr_stem_stopwords_adv_model, cv_stem_stopwords_adv_feats)

    # prediction with lemmatized headlines
    print("--- Lemmatisation Only ----")
    lemmatized_train_data = lemmatize_all_headlines(train_headlines_data)
    lemmatized_test_data = lemmatize_all_headlines(test_headlines_data)
    # print(lemmatized_train_data[0])
    # print(lemmatized_train_data[1])

    # lemmatize only
    cv_lemmatize = CountVectorizer()
    cv_train_lemmatize = cv_lemmatize.fit_transform(lemmatized_train_data)
    cv_test_lemmatize = cv_lemmatize.transform(lemmatized_test_data)
    cv_lemmatize_features = cv_lemmatize.get_feature_names()

    nb_lemmatize_model = train_eval_model(cv_train_lemmatize, train_df['Label'], MultinomialNB(), cv_test_lemmatize, test_df['Label'])
    print_coefficients(nb_lemmatize_model, cv_lemmatize_features)
    lr_lemmatize_model = train_eval_model(cv_train_lemmatize, train_df['Label'], LogisticRegression(), cv_test_lemmatize, test_df['Label'])
    print_coefficients(lr_lemmatize_model, cv_lemmatize_features)
    
    # lemmatize and stopwords
    print("--- Lemmatisation and Stop Word Removal ----")
    cv_lemmatize_stopwords = CountVectorizer(stop_words=nltk_stop_words)
    cv_train_lemmatize_stopwords = cv_lemmatize_stopwords.fit_transform(lemmatized_train_data)
    cv_test_lemmatize_stopwords = cv_lemmatize_stopwords.transform(lemmatized_test_data)
    cv_lemmatize_stopwords_features = cv_lemmatize_stopwords.get_feature_names()

    nb_lemmatize_stopwords_model = train_eval_model(cv_train_lemmatize_stopwords, train_df['Label'], MultinomialNB(), cv_test_lemmatize_stopwords, test_df['Label'])
    print_coefficients(nb_lemmatize_stopwords_model, cv_lemmatize_stopwords_features)
    lr_lemmatize_stopwords_model = train_eval_model(cv_train_lemmatize_stopwords, train_df['Label'], LogisticRegression(), cv_test_lemmatize_stopwords, test_df['Label'])
    print_coefficients(lr_lemmatize_stopwords_model, cv_lemmatize_stopwords_features)

    # lemmatize and adv stopwords
    print("--- Lemmatisation and Advanced Stop Word Removal ----")
    cv_lemmatize_stopwords_adv = CountVectorizer(stop_words=nltk_stop_words, max_df=0.97, min_df=0.03)
    cv_train_lemmatize_stopwords_adv = cv_lemmatize_stopwords_adv.fit_transform(lemmatized_train_data)
    cv_test_lemmatize_stopwords_adv = cv_lemmatize_stopwords_adv.transform(lemmatized_test_data)
    cv_lemmatize_stopwords_adv_feats = cv_lemmatize_stopwords_adv.get_feature_names()

    nb_lemmatize_stopwords_adv = train_eval_model(cv_train_lemmatize_stopwords_adv, train_df['Label'], MultinomialNB(), cv_test_lemmatize_stopwords_adv, test_df['Label'])
    print_coefficients(nb_lemmatize_stopwords_adv, cv_lemmatize_stopwords_adv_feats)
    lr_lemmatize_stopwords_adv = train_eval_model(cv_train_lemmatize_stopwords_adv, train_df['Label'], LogisticRegression(), cv_test_lemmatize_stopwords_adv, test_df['Label'])
    print_coefficients(lr_lemmatize_stopwords_adv, cv_lemmatize_stopwords_adv_feats)
