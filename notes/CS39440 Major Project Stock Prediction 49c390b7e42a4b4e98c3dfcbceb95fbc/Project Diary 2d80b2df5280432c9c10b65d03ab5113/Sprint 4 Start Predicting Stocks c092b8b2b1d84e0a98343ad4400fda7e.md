# Sprint 4: Start Predicting Stocks

# Wednesday — Sprint Planning

## Planning

- Go with the VADER idea, also try the basic stuff like Linear Regression
- Compare Stemming and Lemmatisation
- Conduct a variety of experiments to gain insight before collecting my own data
- Broke down larger issues into sub tasks

## Notes on Trump Project

- Start without removing stop words and such!
- Investigate if removing these actually improve performance or not
- Some good things in there like the overview of the Sprint tasks, topics
    - Should add a table with the goals for each Sprint
- Should I be using nose for testing?

## Work

Linear Regression/Naive Bayes on Dow Jones Data

- Looked at creating packages for this spike, and how they work to get a refresher
- Started work on a file to apply a classifer on the DJIA data
    - Had to figure out how to collate the headline data together for each row
        - Took some time
    - Added function to remove the start and end quotes for the headlines
    - Was not sure whether to go with NLTK or SciKitLearn, leaning towards NLTK, just need to convert the data to the right format: tokens in a dictionary, with the label attached
        - Trying to figure out how to do that best

# Thursday

- Figuring out how to convert the data to the right format for both NLTK and SciKitLearn
- Need to get a better understanding of how SciKitLearn works, including its feature extractors and other stuff.
- Added code for Sklearn and CountVectorizer
- Added NaiveBayes Classification (48%)
- Cleaned up code and add a fucntion to combine the rows for each headlines

# Friday

- Having difficulty using NLTK, because of the format the data needs to be in, and I do not how to convert it to that format and there aren't many examples online to show me how, most examples all use sklearn
- Trying to understand feature extraction
    - Read the start of chapter three in practical NLP, up until Bag of N-Grams
        - Notability Notes
    - Looked at an article on Vector Space: [https://towardsdatascience.com/lets-understand-the-vector-space-model-in-machine-learning-by-modelling-cars-b60a8df6684f](https://towardsdatascience.com/lets-understand-the-vector-space-model-in-machine-learning-by-modelling-cars-b60a8df6684f)
- Will need to look at the different feature extraction, and how that impacts performance too!:
    - Bag of Words
    - Binary Bag of Words
    - Bag of N-Grams
    - TF-IDF
- Going to use sklearn for now, just want to understand what some of the functions do!
    - e.g. CountVectorizer.fit_transform() vs CountVectorizer.fit()
- Coefficients provide the weighting of each feature for positive and negative price movement
    - But are to be deprecated, because it converts Bayes to a linear model, which it is not
    - All of my coefficients were negative
- Confused why feature extraction (e.g. CountVectorizer is needed) used for sklearn, but not in nltk
- Struggling to understand the Sklearn library, particularly things like the fit, transform and fit_transform methods for CountVectorizer
- **NaiveBayes performed better than LogisticRegression 48-48% versus 42-43%**
    - But the LogisticRegression coefficients were a lot more interesting and there was of positive and negative, unlinke for NaiveBayes, which only had negative coefficients and these coefficients were stop words like the, to, of
- Refactoring the code
- Making sure information about first Sprint is retained — started a document:
- Trying to figure out what the Count Vectorizer does?
    - How much pre-processing does it do?
- Count Vectorizer seems to perform tokenization
    - Removing all punctuation, and split words into tokens based on their punctuation, e.g. 'we've' becomes 'we' and 've'

- Read Paper on StopWord Lists in Open-Source packages: [https://www.aclweb.org/anthology/W18-2502.pdf](https://www.aclweb.org/anthology/W18-2502.pdf), [https://www.aclweb.org/anthology/W18-2502/](https://www.aclweb.org/anthology/W18-2502/)
    - Citation BibTex:

        ```latex
        @inproceedings{nothman-etal-2018-stop,
            title = "Stop Word Lists in Free Open-source Software Packages",
            author = "Nothman, Joel  and
              Qin, Hanmin  and
              Yurchak, Roman",
            booktitle = "Proceedings of Workshop for {NLP} Open Source Software ({NLP}-{OSS})",
            month = jul,
            year = "2018",
            address = "Melbourne, Australia",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/W18-2502",
            doi = "10.18653/v1/W18-2502",
            pages = "7--12",
            abstract = "Open-source software packages for language processing often include stop word lists. Users may apply them without awareness of their surprising omissions (e.g. {``}hasn{'}t{''} but not {``}hadn{'}t{''}) and inclusions ({``}computer{''}), or their incompatibility with a particular tokenizer. Motivated by issues raised about the Scikit-learn stop list, we investigate variation among and consistency within 52 popular English-language stop lists, and propose strategies for mitigating these issues.",
        }

        ```

    - When creating my own pre-processing functions, I was unsure how to deal with enclitics
    - Noticed some of these issues during the development of my own preprocessing functions: matching removal of punctuation and tokenizer to the stop word list
- It seems that I do not need to implement my own pre-processing as I thought, it is instead available in the feature extractors in sklearn
- Comment: NLP seems to still be a young, undeveloped field with some of the documentation being hard to understand, and problems like stopwords — no universal approach and connection with tokenizers?
- Want to now apply stopword removal (will probably use NLTK stopwords)
    - Looked at sklearn ENGLISH_STOP_WORDS list and is not as good as NLTK, it may include too many words and does not include 't' 've', which will be tokens after tokenization by the default sklearn tokenizer, created from splitting enclitics
    - Also looked at the newer up-to-date english stop_words list it was harder to find needed to run the code below, I think they are the same because they both contain the same number of words and characters, but it is hard to tell, because the words come in a random order

    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    print(CountVectorizer(stop_words='english').get_stop_words())
    ```

- Project has felt slow at times, lack of direction?
- How do I structure my report?
    - Does Technical Work Section include a chronological discussion of the work completed over the whole project?
    - I think this would make sense, just not sure how to structure it?
        - How do I discuss the research, tutorials and spike work completed, before the actual development of the tool was started?
    - Within the 'Technical Work' section: Do I have chapter on Learning, Research and Spike Work, and a Chapter on the Design, Implementation and Testing of the final tool
    - Individual chapters on Design, Implementation and Testing would not really work anyway when using an Agile approach
        - Want to discuss the individual iterations and the design, implementation and testing within those individual interations? I think, not 100% sure though. Mateusz used three distinct sections even with an Agile Approach
- Been gaining background information as I am going along with the technical implementation of the project — through tutorials or through encountering something I do not understand and need to understand to be able to use that technique, library, etc.
- Sentiment Analysis story sub-tasks may need updating.
- Could possibly add a GUI, with Python back-end server, use Angular for the front-end
    - Docker Container too?

# Sunday

- Research into Multinomial Naive Bayes:
    - [https://www.mygreatlearning.com/blog/multinomial-naive-bayes-explained/](https://www.mygreatlearning.com/blog/multinomial-naive-bayes-explained/)
    - [https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)
    - Overview of several of the Naive Bayes Classifiers/Algorithms:
        - [https://towardsdatascience.com/comparing-a-variety-of-naive-bayes-classificaton-algorithms-fc5fa298379e](https://towardsdatascience.com/comparing-a-variety-of-naive-bayes-classification-algorithms-fc5fa298379e)
- Research into the different Linear Models:
    - Overview of sklearn Linear Models:
        - [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
    - Explanation of Linear Regression Basic
        - [https://towardsdatascience.com/linear-regression-explained-d0a1068accb9](https://towardsdatascience.com/linear-regression-explained-d0a1068accb9)
        - [https://machinelearningmastery.com/linear-regression-for-machine-learning/](https://machinelearningmastery.com/linear-regression-for-machine-learning/)
    - More in-depth explanination of Linear Regression:
        - [https://towardsdatascience.com/linear-regression-explained-1b36f97b7572](https://towardsdatascience.com/linear-regression-explained-1b36f97b7572)
    - Logistic Regression Explanations:
        - [https://towardsdatascience.com/logistic-regression-explained-9ee73cede081](https://towardsdatascience.com/logistic-regression-explained-9ee73cede081)
    - Logistical Regression for Sentiment Analysis
        - Article: [https://towardsdatascience.com/sentiment-classification-with-logistic-regression-analyzing-yelp-reviews-3981678c3b44#c3b8](https://towardsdatascience.com/sentiment-classification-with-logistic-regression-analyzing-yelp-reviews-3981678c3b44#c3b8)
        - Code: [https://www.kaggle.com/dehaozhang/sentiment-analysis-with-lr](https://www.kaggle.com/dehaozhang/sentiment-analysis-with-lr)
    - Differences between Linear and Logistic Regression:
        - 

## Different Bayes Algorithms

The Naive Bayes model encompasses the idea of treating the features as independent but does not describe the distribution of features. Whereas Multinomial Naive Bayes, Gaussian Naive Bayes and others describe the distribution of features. A Gaussian Naive Bayes classifier assumes that a normal (or gaussian) distribution, which can be used for continuous data. Multinomial Bayes describes a multinomial distribution, a distribution with a fixed number of outcomes two or more, which all have an equal probability of occurring. E.g. rolling a dice with six sides, all six possible outcomes have an equal chance of happening each time the dice is rolled: 1/6. 

## Linear Models

Linear Regression plots a line that best classifies the features in terms of the target value. in terms of sentiment analysis, it can classify what words are the most positive and what words are the most negative and everything in-between. Not sure how it would classify text as positive or negative though. Good for a continuous medium e.g. height

Logistic Regression, based on the theory of Linear Regression, but instead of the Regression model it is a classification model. A Sigmoid curve is used to separate the data. Provide probability based on feature values, uses that for classification? Can only be used for binary classification. 

Dicriminative vs Generative Models

- Generative: naive bayes, hidden markov models
- Discriminative: logistic regression, SVMs

- Could be using predict() in sklearn and then the metrics library for accuracy and F1 score metrics
    - Would involve a bit too many changes right now, maybe future
- Creating confusion matrix?

- Getting confused on how sklearn.CountVectorizer() should be used, passing it the nltk stopwords does not work out of the box
    - Was not an issue with CountVectorizer, I used fit() on training data instead of transform()
- **Naive Bayes accuracy slightly improves after stopword removal, and the most positive features are a lot more interesting — they are no longer simply stop words**
- **Logistic Regression accuracy dwindles, and top features do not change significantly**
- How to apply lemmatisation or stemming with sklearn?
    - strip_accents? — not sure what that does exactly
    - Maybe replace the pre-processor
    - Seems a lot more difficult than stop word removal
    - Maybe need to understand the Pipelines?
- Sklearn terms:
    - X_train/test: train/test features e.g. text, bag of words
    - y_train/test: train/test labels for features e.g. positive, negative
- Just apply Lemmatization before passing the text onto the CountVectorizer
- Maybe in future I should apply all pre-processing manually myself, before even using the CountVectorizer?
    - Can I pass the CountVectorizer tokenized text?
        - Don't think so, but could just detokenize the data, before passing it to the CountVectorizer
    - Utilising the CleanText class, with fit_transform approach used in the Airline Tutorial
- Pipelines do not seem to be the answer
- Need to evaluate how to proceed
    - Do not think want to work on pre-processing story, before the next Sprint
- Using the sklearn and NLTK libraries correctly can sometimes be difficult
    - sklearn works pretty differently, having tokenizers and pre-processing in the CountVectorizer
        - Does not happen beforehand like with NLTK
    - NLTK's input to the classifiers is strange compared to sklearn, and involves a lengthy conversion process to get to the required format
        - I do not currently understand the point of the dictionary marking each feature with the value 'True'
    - Was not sure which one to use, kind of jumped between them depending on the specific scenario: for DJIA data I had examples of how to use sklearn on that data, but not NLTK with pandas data frames, so I went with sklearn over NLTK
- Learning about NLP, Python, the libraries has been the most time-consuming part of this project
    - Think it was still a good choice to go with Python over another language
- If I pre-process the data before passing it to the count vectorizer then I could also use NLTK easier too?
- So I think I want to start pre-processing the data, before passing it to the CountVectorizer
- Start a new spike, file possibly?

# Monday

- Daily stand-up
    - Removed story to create a production environment FMP-126
    - Instead, focus on the Sentiment Analysis Spike FMP-98, use that knowledge to create a productive application next Sprint
    - Should be on track to complete the rest of the stories in the Sprint!
- Still not sure about Python package structure and imports
- You would think using the same framework would be ideal, but not necessarily, when the tokenizers and other aspects do not match their stopword lists and other things.
- Found out some info about stemming
- **Look at Notability notes on design for the current Spike**
- Documentation Gap NLTK — how do I use the WordNetLemmatizer with POS tags?
- No documentation on wordnet corpus
- Relying a lot on examples, tutorials, not sure where they learned how to use the library. Looking at the sources?
- TDD not applied much during spikes, if tests written, used test-last development techniques
    - Because I was unsure how pre-processing fuctions should or might work, I went stratight into experimentation first
- Hard to figure out what is best, just need to go with something good for now
- Going to pass pre-processed text with punctuation removed to lemmatizer for now, may change that in future
    - Get something simple twek later
    - Porblem, may be I am trying to be too perfect from the get go
- Current pre-processing problems:
    - 'U.S.' gets removed completely, by sklearn and similar tokenization
    - Stopwords like 'not' get removed
- Need to investigate the pre-processing further in the future
- Data most important aspect, so spending a lot of time on processing the data and how I go about that processing
- Looked at Mateusz's report again for some perspective, ideas
    - He did mark his tweets with sentiment

- Maybe I do not need to mark the data with a sentiment label?
    - Just analyse what impacts the stocks the most?
- Does positive sentiment analysis equal positive price movement
    - Train a sentiment analyser and then train it to link the sentiment to the price movement

- Still figuring out how to best use stories and tasks
    - especially when I am unsure of the work to be completed

## Applying Stemming and Lemmatisation + Advanced Stop Word Removal

- Naive Bayes: stemming alone improves performance by 0.2 (2%) over no-preprocessing, and 0.1 over stop word removal
    - Top postive features still all stop words
    - Top negative features more indicative like before
- Naive Bayes: stemming & stop word removal improves performance by 0.1 over no-preprocessing, but is not as good as only stemming
    - Top positive features are more information though, not purely stop words
    - Strange result, means stop words somehow correlate with the positive days?
        - longer headlines?
- Naive Bayes: Advanced stop word removal reduces performance significantly, 0.5 over the baseline and 0.5 over basic stop word removal
- Naive Bayes: stemming & advanced stop word removal, drops performance by 0.06 vs baseline, and not as good as only stemming
- Naive Bayes: lemmatisation 48.6% worse than baseline by 0.1
- Naive Bayes: lemmatisation & stop word removal 48.1%: worse than baseline by 0.15

- Is this performance down to randomness or something explanable?

- sklearn did not provide stemming or lemmatisation through the CountVectorizer, which I wanted to apply to the text and experiment with that
    - Had to figure out when to do the lemmatisation, after reading the Practical NLP book I realised before was better
    - Had to figure out how and when to tokenize the text
    - Was I going to use all my own pre-processing or still use sklearn's CountVectorizer built-in functions
    - Had to figure out how to implement lemmatization and stemming in an acceptable way
        - How to use the libraries
        - Which stemmers and lemmatizers to use
        - Went for NLTK ones: WordNet and Porter
            - Porter was apparently the least aggresive, and I seen it used in the tutorials
        - Test all the pre-process functions to make they were working as expected
            - Check the code worked as expected in general

# Tuesday

- Daily stand-up is when I make/review pull requests
- Daily stand-up
    - Need to merge yesterday's work
    - Going to work on corrections first, because I know how much of that I can actually complete
- Need to add some more READMEs to the repo to explain the different folders
- Worked on updating the methodology
- Notability Notes on this Spike
- Need to look for the specific citations, e.g. for VADER:
    - Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

- Vader analysis
    - Sentiment scores show all training articles are all very negative in sentiment
    - Not sure how to get the sklearn models to use non text data
    - ASk Neil about his?
- Next Print: build production ready application for DJIA dataset
    - Do more tweaking, use n-grams?
    - Use TFID vectorizer
    - Vader Analysis?

# Wednesday

## Notes Trump Sentiment Analysis

- Get test accuracy?
- Build my own tokenizer? with RegEx tokenizer NLTK or build it from scratch
- Cross Validation, K-fold should be added
- Do some Text Analysis to improve accuracy in future?
    - Investigate impact of stop words, negative stop words, lemmatisation, stemming, punctuation etc.

## Retrospective, Review

- FMP-98 Apply Sentiment Analysis to Stock Market Specific Dataset was underestimated, involved a lot more learning about sklearn and the classifiers than expected
- Sometimes difficult to stay motivated, becuase I keep coming accross unexpected problems and new things I need to learn, but this keeps it challenging and means I keep learning new things and get to solve new problems
    - Need to try and stay motivated, even when I do come across issues like this!
- Still missing knowledge in several NLP areas, I think PracticalNLP book can help with these gaps and provide guidance, if I have a chance to continue reading it