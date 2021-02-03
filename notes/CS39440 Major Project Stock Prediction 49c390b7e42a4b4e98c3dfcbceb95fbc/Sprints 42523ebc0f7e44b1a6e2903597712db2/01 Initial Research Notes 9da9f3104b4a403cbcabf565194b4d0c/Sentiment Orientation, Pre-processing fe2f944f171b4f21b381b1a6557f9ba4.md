# Sentiment Orientation, Pre-processing

# Preparing the Data

- Clean the text data, remove stop words and stemming
- Evaluation methods/metrics: precision, recall and the F1 score.

- Aspect-based sentiment analysis: not just positive or negative, but more specific domain phrases: blurry, inexpensive, poor

- Simplest Approach: Two Dictionaries: one with positive and the other with negative words
    - Not a machine learning approach, useful when there is not a rich-enough labelled data set
    - Will run into issues sooner or later, machine learning approach will be better in the long run
- Can still use a dictionary in an ML approach, use it as features?
    - Add dictionary value, sentiment pairs to training data (e.g. not good, negative)

- Lemmatization and stemming convert words to their orignal form removing grammar tense
    - Lemmatization - looks more readable and interpretable than stemming

# Stock, Headline Dictionary

The subjects, e.g. countries at war, would be relavant in analysis

Negatives:

- Brink of war
- Increase in unemployment
- Impeachement, about to be impeached
- Invasion, War
- Earthquake, natural disaster, hurricane