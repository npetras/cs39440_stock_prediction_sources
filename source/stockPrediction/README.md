# Stock Prediction Application
# Installation
* Need to download the following data from nltk:
    * wordnet
    * averaged_perceptron_tagger
* This can be done so at the command line with the following:
```
$ python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```


# Packages
## main
The main sources for the program, contains all non-test sources.
## main.data
Contains all sources related to processing the data, including cleaning, analysis and 
representation of the data.
## test 
Contains all the tests for the main sources in the main package. 