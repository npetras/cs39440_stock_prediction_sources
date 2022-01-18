# Sprint 3 - Sentiment Analysis Practice & Background Research

# Wednesday - Sprint Planning

## Sprint Planning

Notes: 

- Continue with work on DJIA dataset — sentiment analysis, then the prediction of price movement
    - Produce classifier that can do at least basic sentiment analysis (positive, neutral and negative), and price prediction
- Look into Background for the Project — how other people used the Dow Jones Dataset
    - After I have completed the Sentiment Analysis
- Focus is on Sentiment Analysis
    - Background also part of the Sprint Goal
- 8.5 Story Points, 7.5 completed in last sprint, using that number as guide, and see 8.5. Particulary the stories being worked on as doable
- Plan to get the work done has been partial devised
    - Will develop further during the Sprint
    - Was not sure what exactly I would be looking into in terms of the Sentiment Analysis, and Background, but had a basic plan: basic Sentiment Analysis Tutorials, looking at books on NLP and Sentiment Analysis. Focus was first to complete the Sentiment Analysis and Methodology write-up before exploring formal background.

## Work

- Punctuation removal is removing 'U.S.' with 'U'
    - Punctuation removal has to be changed to remove only apostrophes with spaces?
- Started looking for tutorials on NLTK's sentiment analysis
    - Digital Ocean
    - NLTK Book

### NLTK Book, Chapter 1

- concordance() - occurance of a give word in some context
- similar() - words that are used in similar contexts
- common_contexts() - examines the contexts shared by two or more words
    - These words should be similar() or nothing may be returned
- token: sequence of characters treated as a group, e.g. hairy, his or :)
- sets can be used to discover the number of unique tokens
- word type or type — distinct words and punctuation, independent of specific occurrences in text
    - word types - when only words considered, is this the lemmas, base words?
    - types - when words and puntuation considered
- NLTK has functions for the frequency distribution of words
- collocations: sequence of words that occur together inusually often, e.g. red wine
    - function available too
    - essentially frequent bigrams
- bigrams: word pairs
    - bigrams function available

Chapter 3, Tokenisation

- variety of tokenizer in NLTK
    - Maybe worth taking a look at which one is best for my application?
        - Work Tokenizer or another one?

- NLTK book was useful but not really want I want — Sentiment Analysis may look at it again in the future though

# Thursday

Stand-up

- Start reading over the 'Practical NLP Book' after NLTK book code is reviewed and merged

- NLTK Book may ne worth looking at again for preprocessing?

## Practical Natural Language Processing Book, Chapter 1

- Explains the basics of language, which I could draw on
- Discusses the different types of approaches to the problem:
    - Heuristic, Machine Learning, Deep Learning
    - Can use this to help me decided and justify my decision for a particular classifier
- Read up to the Deep Learning in the Pratical NLP Book

- Definetly a useful resource from which to draw some background

## Digital Ocean Tutorial

[https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk#prerequisites](https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk#prerequisites)

- Normalising data — turning it into its canonical form — stemming and lemmatisation
- Pos tagging — lots of different tags, will I have to learn, explain these?
    - [https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
- Noise removal - removing parts of text that do not add meaning or information to the data
    - Punctuation may provide context, but this context is hard to process
- Text Analysis
    - Word Density, FreqDist
- Completed up to step 6 of the Digital Ocean Tutorials

Future Tutorials for Sentiment Analysis:

[https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386](https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386)

[https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk](https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk)

[https://www.datacamp.com/community/tutorials/simplifying-sentiment-analysis-python](https://www.datacamp.com/community/tutorials/simplifying-sentiment-analysis-python)

[https://realpython.com/python-nltk-sentiment-analysis/](https://realpython.com/python-nltk-sentiment-analysis/)

# Friday

- Referencing Videos
- Looked at using Mendeley, but it does not export in the desired BibTeX format, unfortunately, but it is a nice tool
- Working on the Methodology write-up:
    - Checking through the text, re-writing some parts
- Reading Chapter One of Practical NLP
    - Deep Learning Section
    - Why Deep Learning is not the Silver Bullet?
- Methodology Write-up:
    - Added new content on why ideal days were used for estimation

# Saturday

- Thinking a little bit about the productive Python application: CI build, testing, package structure, dependency management etc.
    - Package Structure: [https://docs.python-guide.org/writing/structure/](https://docs.python-guide.org/writing/structure/)
    - Testing your code: [https://docs.python-guide.org/writing/tests/](https://docs.python-guide.org/writing/tests/)
- Hitchhiker's Guide to Python (free online version): [https://docs.python-guide.org/](https://docs.python-guide.org/)
- CI Build to run on all pull requests and on the master branch too
    - Preferably pull request cannot be merged until CI build passes
- Writing up some notes about the XP Practices
- Creating stories and sub-tasks from above investigation, discussion ^, Backlog Grooming/Refinement
- Thinking about where to write up implementation different parts of the report, especially specific practices
- Skimmed Extreme Programming Explained book, practices in articles do not match the Extreme Programming book

## Digital Ocean Tutorial

- NLTK has its own classifiers, do not need to use sklearn library

# Monday

## Methodology Notes

- Methodology point on XP practices mentioned online differing to the book:
    - XP over Scrum:
        - Scrum is simpler, more flexible, and community is a lot more active — the methodolgy is frequently refined and updated by Schwaber and Sutherland
        - Last official writing on XP was released in 2004 by Beck
        - Some disagreement between Kent's writing and the description of the framework online
    - XP Practice Usage:
        - I will be using the development practices originating from or inspired by the XP framework and community, mentioned in Alexasoft, apart from the planning game, pair programming, and collective code ownership
            - 40-hr week?
        - I am not strictly following the Extreme Programming methodology, but do see value in many of the practices the framework enstilled in the industry
            - Most of, if not all of the practices mentioned in the Alexasoft article are seen as  good practice through the software industry

- Mention general approach to work:
    - Technical Work
    - Research
    - Documenting my work everyday
    - Writing up parts of the report as I went along, while my thoughts on those topics were fresh

## Sentiment Analysis

- Good idea to focus on simpler libraries: NLTK over SciKitLearn
    - NLTK is a lot simpler and easier to use, and more intuative than SciKitLearn

### TowardsDataScience NLTK Binary Sentiment Analysis

[https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386](https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386)

- Can use SciKitLearn along with NLTK, scikitclearn module in NLTK can inherit the proerties of an SciKitLearn classifier
- SciKitLearn provides metrics, what about NLTK?
- EnsembleModel use the predictions from multimple models and choose the majority prediction
- Can pickle trained classifiers, since they can take a long time to train

### DataCamp: Text Analytics

[https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk](https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk)

- Bag of Words (BOW) model
    - Not taking into account the context of the text
    - Will likely be my first approach to the problem
- Word Tokenizer breaks the text into works, sentence tokenizer breaks it into sentences
- Does SciKitLearn provide many more classifiers?
    - How many are there in NLTK?:
    - Not as many, a few examples
    - NLTK NaiveBayes may be a good first approach or baseline, but SciKitLearn along with NLTK may a good idea for a larger variety and more sophisticated classifiers
- What are pipelines?
- Sentiment Analysis, two main approaches:
    - Dictionary, Lexicon-based -- count the number of positive and negative words
    - Machine Learning — trained using pre-labelled data
- Bag of Words is the simplest form of feature extraction
    - Converts text into matrix of occurances of words within a document
- CountVector is Bag of Words?
- What is X_train, X_test, y_train, and y_test?
- TF-IDF (Term Frequency - Inverse Document Frequency)
- Look into feature extraction a little more!

[https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/](https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/)

[https://realpython.com/python-nltk-sentiment-analysis/](https://realpython.com/python-nltk-sentiment-analysis/)

[https://www.nltk.org/book/ch01.html](https://www.nltk.org/book/ch01.html)

[https://bbengfort.github.io/2016/05/text-classification-nltk-sckit-learn/](https://bbengfort.github.io/2016/05/text-classification-nltk-sckit-learn/)

[https://www.nltk.org/book/ch06.html](https://www.nltk.org/book/ch06.html)

# Tuesday

## Data Camp: Simplifying Text Analysis

[https://www.datacamp.com/community/tutorials/simplifying-sentiment-analysis-python](https://www.datacamp.com/community/tutorials/simplifying-sentiment-analysis-python)

- Corpus, Dictionary: can be created from Bag of Words

- I plan to use NaiveBayes from NLTK for now, not really planning on using SciKit-Learn until later on when I get the first classifier working
- Multinomial: special type of probability distribution

## Background Research DJIA Dataset

Notebook 1: OMG! NLP with the DJIA and Reddit

[https://www.kaggle.com/ndrewgele/omg-nlp-with-the-djia-and-reddit](https://www.kaggle.com/ndrewgele/omg-nlp-with-the-djia-and-reddit)

- Using CountVectorizer, TfidVectorizer for feature extraction
- Use LinearRegression
- Improves performance by using bi-grams, instead of just single words
- Feature Extraction CountVectorizer: produces 'table' with counts for each word
    - Removes duplicates and replaces them with counts

Notebook 2: Stock Price Prediction - 94% XGBoost

[https://www.kaggle.com/shreyams/stock-price-prediction-94-xgboost](https://www.kaggle.com/shreyams/stock-price-prediction-94-xgboost)

- Lot more complex, used some algorithm to generate sentiment analysis and other scores for each row of headlines

Notebook 3: Use News to predict Stock Markets

[https://www.kaggle.com/lseiyjg/use-news-to-predict-stock-markets](https://www.kaggle.com/lseiyjg/use-news-to-predict-stock-markets)

- Logistical Regression
- Improves performances by going from singular words with noise to bi-grams with frequent and in-frequent words removed

Notebook 4: News Headlines Stock Sentiment Analysis

[https://www.kaggle.com/pkmisra/news-headlines-stock-sentiment-analysis](https://www.kaggle.com/pkmisra/news-headlines-stock-sentiment-analysis)

Notebook 5: Predicting stocks(up or down) 87% accuracy!

 [https://www.kaggle.com/nitishkumarpilla/predicting-stocks-up-or-down-87-accuracy](https://www.kaggle.com/nitishkumarpilla/predicting-stocks-up-or-down-87-accuracy)

# Wednesday - Review and Retro

## Review

- Everything done, apart from part of the Background research
- Reduced Velocity - interestingly
- Good progress, met Sprint Goal, just not sure what to do in the upcoming Sprint

## Retrospective

- Have a look at story points/velecity for next Sprint, what is affecting the large swing?
- Maybe Sentiment Analysis story was underestimated?
    - Included a lot of background research too in the end, maybe should not have, should have been delegated to the Background Task
- Methodology Issue also underestimated?
- Progess being made, ready for implementing Sentiment Analysis on Stock Market Data
- Need to get the code figured out for the Stock Market Prediction now!

- **Forgot that Sprint 2 Numbers were bloated!!!**
    - Some Rejected Story points got counted!