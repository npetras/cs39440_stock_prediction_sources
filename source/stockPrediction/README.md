# Stock Prediction Application
# Installation

## Method One - requirements.txt
Make sure you have python and pip installed and use this command on the command line (in this directory) to install the required packages:
```
pip install -r requirements.txt
```

The following command is also required to download the necessary data from nltk: wordnet and averaged_perceptron_tagger:
```
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

## Method Two - pipenv
If you install pipenv using the following pip command, after you have already installed python and pip:
```
pip install pipenv
```

You can use pipenv to install the required dependencies like so, if you are inside this directory:
```
pipenv install
```

Like for method one the following command is also required to download the necessary data from nltk: wordnet and averaged_perceptron_tagger:
```
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

# Packages
## main
The main sources for the program, contains all non-test sources.
## test 
Contains all the tests for the main sources in the main package. 

All of the packages in test should be a reflection of the same packages in main, with a test module and class for each module and class in the main package.
## main.data
Contains all sources related to processing the data, including cleaning, analysis and 
representation of the data.
## main.classifier
Contains sources related to running classifiers on the data to predict price movement. 
