# Sprint 05: Create Productive Application

# Wednesday

## Planning

- Want to get started on the productive application, and continue some of my learning/background so I can create a better program
- Need to look at more advanced feature extraction soon!

## Work

- Started working on creating the production project, using this guide:
    - Dependency Management/Virtual Environments: [https://docs.python-guide.org/dev/virtualenvs/](https://docs.python-guide.org/dev/virtualenvs/)
        - pipenv vs virtualenv: [https://medium.com/@krishnaregmi/pipenv-vs-virtualenv-vs-conda-environment-3dde3f6869ed](https://medium.com/@krishnaregmi/pipenv-vs-virtualenv-vs-conda-environment-3dde3f6869ed)
    - Structure: [https://docs.python-guide.org/writing/structure/](https://docs.python-guide.org/writing/structure/)
- Created basic folder structure for Python Project created in PyCharm
- Can run a directory with tests in it with an command
- Added a CI build and rule to prevent PRs being merged before they pass the Travis build
    - CI build includes:
        - test run
        - pylint
- Using yapf for formatting the Python code

# Thursday

- Difficult to manage learning and implementation, want to have something tangible, but also want be confident in what I am doing
- Working on the desing of the productive application, look at notability note for more information

# Saturday

## Classifier Performance

- For recording and for the Poster

[Naive Bayes (MultinomialNB)](Sprint%2005%20Create%20Productive%20Application%2020e83212965747ca8a37c040cbf1b1fa/Naive%20Bayes%20(MultinomialNB)%20c4d206f2533a4b13926098deafcd44a5.csv)

[Logistic Regression](Sprint%2005%20Create%20Productive%20Application%2020e83212965747ca8a37c040cbf1b1fa/Logistic%20Regression%2069133c8919eb4dd4b7bdef1b19a22c58.csv)

- Result above, may indicate that at least for logistic regression, only removing the most frequent and most infrequent words may be the best strategy to removing 'stopwords'
    - Since simple stopword removal reduces the accuracy of logistic regression
    - But advanced stop word removal improved accuracy
- Stemming more useful for this data?
- Need to apply cross validation, before properly evaluating!

- Using the spike code, along with my new designs to create the production code
    - Using TDD for the new code and code that is being reworked
- Used the structure from maven to split up main sources (non-test code) and test code
    - So they can be treated differently
        - Use a script to run all tests in the test folder
    - Maybe should remove the main package and just put [main.py](http://main.py) and the other folders into the top-level folder alongside tests
- Software Engineering takes time
    - Commenting, tests, style, thinking through your decisions, etc.

# Sunday

- Trying to figure out how to run test code, outside of the project directory for the CI build
    - Currently breaking with the error:

        ```python
        ModuleNotFoundError: No module named 'main'
        ```

    - Been trying a variety of fixes to the problem:
        - Using [context.py](http://context.py) did not seem to fix the problem, still get:

        ```python
        ValueError: attempted relative import beyond top-level package
        ```

    - Had to specify some extra options for the python unittest command - start directory and top-level project directory
    - No having problems with missing data from nltk
        - Installed in the CI script
        - Better to do it in code? Probably not, will package up the code anyway
            - Just may be harder for someone else trying to run my code.
- Correcting the lint errors highlighted by pylint
- Better to download nltk on command line or in code?
    - Doing on command line, becasue that is the best practice I have seen online
        - Everywhere shows to install it on command line, nltk reccomends to install it on command line
    - Probably not want to slow down code, adding that command, which will be required to run on every execution of the file

    # Monday/Tuesday

    - Did little to no work
    - Still unmotivated from stress, from feeling overwhelmed by the amount of work required to complete the project to a good standard and overloaded and overworked with current work
        - Feel like I am failing slightly, been putting in lots of work over the past weeks, but still lots to do
        - Do not feel particularly on track
    - Currently using hold-out, based on the recommendations in the DJIA dataset

    # Wednesday

    ## Review/Retrospective

    - Re-focus on the next Sprint
    - Want to continue with work on the report in the upcoming weeks