# NLP, Sentiment Analysis Experimentation

# Alexa Review Example

[https://towardsdatascience.com/cleaning-preprocessing-text-data-for-sentiment-analysis-382a41f150d6](https://towardsdatascience.com/cleaning-preprocessing-text-data-for-sentiment-analysis-382a41f150d6)

## Libraries

- re: regular expressions, built-in python library
- pandas: data analysis and manipulation tool
- nltk: natural language toolkit
    - NLP library
- wordcloud: creating word clouds in Python
- spacy: NLP processing library
    - Needed to download spacy and the English (en) pack
- numpy: scientific computing
    - efficient arrays
    - array operations
    - built in C — very fast, much more efficient that Python operations
- pickle: serialisation Python objects - Python library
- seaborn: statistical data visualisation, based on matplotlib
    - creating graphs
- matplotlib: data visualisation
- sklearn: machine learning, predictive data analysis
- gensim: NLP, topic modelling
- vaderSentiment: lexicon and rule-based sentiment analysis tool
    - specifically attuned to social media, and works well on other domains

## Terminology

- corpus/corpora:
    - collection of written texts
    - body of knowledge or evidence
    - especially the entire works of a particular author or a body of writing on a particular subject
    - In NLTK contains list of stop words which can be removed — not impact on sentiment analysis

- topic modelling: statistically modelling used for discovering the abstract topics that occur in a collection of documents
- Latent Dirichlet Allocation (LDA)
    - Maps documents to a distribution of topics
    - Unsupervised learning technique for topic modelling