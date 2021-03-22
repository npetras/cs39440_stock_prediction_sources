# Sprint 07: Improving DJIA Prediction, Tesla Data Collection

# Wednesday

- Look at previous Sprint for Planning
- Figuring out how to use Cross Validation
    - [https://scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)
    - [https://towardsdatascience.com/complete-guide-to-pythons-cross-validation-with-examples-a9676b5cac12](https://towardsdatascience.com/complete-guide-to-pythons-cross-validation-with-examples-a9676b5cac12)
- Easiest way to do it cross_val_score()
- Other options like KFold and other more manual techniques
- Run module, was initially designed with the use of different text representations in mind. Since CountVectorizer is an object of sklearn and perform preprocessing on the data, it was assumed that TfidVectorization would behave in the same or similar way
    - Was not sure how to structure this initially, maybe can be improved
- Lot of complexity, new terms being thrown at me
    - Takes a lot of brain power to try and understand how things work and how to apply them in code, requiring frequent breaks
- Implemented Cross Validation was easier than I expected, but lost the ability to print features/co-efficients now
- Thought about mixing NLTK and sklearn with the SklearnEstimator or similar object in NLTK, but thought it may just complicate things further, so I stuck with the sklearn library for modelling, classification etc. and used NLTK only for the preprocessing
- sklearn estimators (including classifiers), take in 2D arrays of features (vectorized input)
    - they do not see the feature names or original values, just numbers that represent them

coef_() method for NaiveBayes

- The coefficient in this case probably represents a factor that measures how impactful each feature is on the classification
- Used to map the model as a linear model
- Replaced with another function, I should use from now on feature_log_probability_
    - Not much documentation, if any documentation on how to use it
- Did not really understand what the coef_ function was and that it only related to linear models, so I missed that feature_log_probability should be used instead. Misreading the description of the coef_ function. It worked so I just used in the first version.
    - Did not understand and still don't understand how the coef_ function even works

- Struggling with the work, when I don't understand how certain aspects work
    - So many matehmatics terms used I am getting so confused and do not understand what is going on
    - Constant problem solving, information overload
- Think I will be using the full name of the module's instead of the shortened versions to make it more explicit what is being called
- Each classifier has to be treated differently
    - Naive Bayes does not act like Logistic Regression and vice-versa

# Thursday, Friday, Saturday

- Not sure if the features being printed for Naive Bayes are correct
- Results with Cross Validation for Naive Bayes are lot more like was expected:

[Naive Bayes](Sprint%2007%20Improving%20DJIA%20Prediction,%20Tesla%20Data%20Co%209bdb170dc3554786b09a23bd1bd19acc/Naive%20Bayes%20db5fc931d6574b27af4962077c25a3d9.csv)

- Improve the general accuracy slighty, over previous best results
- There most have been some inherent pattern that the classifier overifit to in the training data when using the hold-out method
- Lemmatisation is a lot more on par with stemming accuracy
- Stop word and frequency removal helps improve accuracy
- Best test accuracy results when train accuracy is the lowest
    - Indiciates a more generalised classifier that is not overfitting to the training data
- F1 Score higher than accuracy
- Best performance with all preprocessing applied

[Logistic Regression](Sprint%2007%20Improving%20DJIA%20Prediction,%20Tesla%20Data%20Co%209bdb170dc3554786b09a23bd1bd19acc/Logistic%20Regression%207fe7daf5b84d442cbe340a8d3ba17687.csv)

- Stemming Only provides better accuracy for Logistic Regression than Stemming, Stopword, Frequency Removal, interestingly
- Lemmatisation Only accuracy is even better than Stemming Only, providing the best overall accuracy
- Lemmatisation alone performs the best
- Stopword removal does not impact the accuracy negatively consistently though
    - Stemming & Stopword better accuracy than Stemming only
    - Lemmatisation & Stopwrod slightly worse accuracy than Lemmatisation Only
    - Stopword alone impacts accuracy negatively
- Frequency Removal Seems to affect performance negatively
- Logisitic Regression seems to fit perfectly to the training data

- Seen that cross_validate() offer several scoring metrics, so I added the f1 score and training scores
    - BEcause I wanted to do this anyway, and it was easy to add
    - Went to cross_validate from cross_val_score, becuase I wanted the top features

    # Friday

- Researching how to do text mining, sentiment mining to get the sentiment from the text without labels
- [https://realpython.com/python-nltk-sentiment-analysis/](https://realpython.com/python-nltk-sentiment-analysis/)
- [https://monkeylearn.com/blog/text-mining-sentiment-analysis/](https://monkeylearn.com/blog/text-mining-sentiment-analysis/)
- [https://www.datasciencecentral.com/profiles/blogs/text-mining-and-sentiment-analyses-a-primer](https://www.datasciencecentral.com/profiles/blogs/text-mining-and-sentiment-analyses-a-primer)
- SentiWordNet - Dictionary that can provides postive, negative values for words

- Text classification difficult could be mentioned in the Background section
    - Middle of the pack

# Saturday

- Doing some more research cannot find anything else that is useful

- Found FinBERT, which seems interesting, but not very useful for me
    - [https://github.com/ProsusAI/finBERT](https://github.com/ProsusAI/finBERT)
    - May be useful to compare performance
- Cross Validation Results Recording and Analysis, filling out above tables^
- Could possible load the feature from Dictionaries
    - When applying the Sentiment Analysis Calculations
- How do I combine the VADER scores for all headlines?
    - Just positive negative counts first?
    - Then do some advanced calculation combination?
        - Add together and divide?
- Was too difficult to add top features to cross validation, so it was not added for now
- Did not make sense to refactor the code either at this point, did not see how to combine the holdout and cross validation functions

# Sunday

- Using
- Should I add logging?
- Using short imports fpor my own modules and ull imports for third party libraries
- Logging not easy to use, skipping it for now
    - Was not able to easily log and enable DEBUG output
    - WARNINGs were still output
- Using these headlines for sentiment is problematic, because they are all negative
- My lemmatization may not be as effective, beccause I am removing punctuation with tokenizer
- Before and after lemmatization, stop word removal and sklearn tokenization, nearly every headline is seen as negative

# Monday

- GUI Frontend designed with Angular, everything running in Docker containers
- Python Web Backend, or simply scripts run in Kotlin?
- Daily Stand-up:
    - Work on collecting my own Tesla data
    - See notability Notes 18 Tesla Data Collection
- Decision was made to focus on News Headlines from reputable sources, instead of using social media posts, because New Headlines are much better quality
    - Unlikely to have typos
    - Every headline will likely be informative
    - Tweets from random users may not be reliable
    - May require lots of filtering, sifting, preprocessing for social media posts

- Scraping Tutorial: [https://towardsdatascience.com/web-scraping-news-articles-in-python-9dd605799558](https://towardsdatascience.com/web-scraping-news-articles-in-python-9dd605799558)
- Very difficult to scrape websites using JavaScript, was not able to find an good easy enough way to do it, I will start collecting the data manually
- Collecting data manually was very time consuming, so I looked if scraping was possible once again, and found a tutorial that allowed me to scrape the websites
- Thought I would not be able to scrape websites using a Script, but after seeing the tutorial below I was able to scrape the website properly using the framework of a different request. Instead of requests, this tutorial uses urllib. When using requests only, I could not access the content I wanted that was obviously there, maybe because the dynamic content created with JS was not displayed from the basic request, or I was using the BeautifulSoup and requests frameworks wrong
    - [https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638](https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638).
- Although I am still only getting the top news headlines for the last few days from Barrons
    - Need to be able to scroll the page, and scrape a much larger set of headlines for at least a few months, preferably a year