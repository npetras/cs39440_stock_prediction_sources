# Sprint 06: Productive Application 2, Mid-Project Demo

# Wednesday

## Planning

- Continue with production application
- Cross-validation required in future
- Better feature extraction
    - Sentiment analysis
    - Mapping positive/negative sentiment to positive/negative price movement or the opposite
        - Counting positive words
        - Counting negative words

- Kept the story point count low, we'll see how it goes and possible add add story/iesortask(s)after the demo

## Work

- Created slides for mid-project demo

# Friday

- Not sure how to create or deal with exceptions in Python yet
- Setting max iterations, even up to 1000 does not improve performance
    - but it does prevent the warning, when set to 250
- Continued work on the productive application, can now run a model with parameters, coefficients are printed
- This error again, when running tests:
    - ImportError: Start directory is not importable: 'source/stockPrediction/test'
    - Fix: command was wrong for the directory it was running in
- Need to update path resolution, make it independent of where the file is run from

# Saturday

- Have not been using formatter much, just been relying on pylint and fixing errors manually
    - Should start using it more often
- First CI builds were running for 30s, now they run over 1 min
    - Still very short
- Could I apply the formatter automatically in the CI build?
- Preparation for demonstration
    - Finishing off first version productive application
        - Created a script to run for the second marker

# Sunday

- Added exception test
    - Did not know how to test standard output and do not see it as worthwhile right now
- COnducting a review of my code, prompted me to make several considerations:
    - Add tests for the run module
    - Update the READMEs
    - Make corrections in the READMEs
- Complete production application work
- Working on starting the Technical Work Overview, Sprint 1
- There has been a bit of disconnect with what I am doing and my supervisor's ideas of what I should be doing, keep getting confused and doubting my plans
- Prep for the demonstration
    - Rehersal

# Monday

Demonstration

- Evaluate your classifer, model versus other exsiting models
- Use more advanced classifiers SVMs

- Reading NLP Chapter 2 — See notes in Notability

Results of Experiments:

- Naive Bayes seems to be fitting to the frequency of words — likely overfitting and not a good general classifier
- Logisitic Regression provides much more informative features
    - Would likely perform better in the general case

## Critical Evaluation

- Good methodology application for structuring my work
    - too much focus on this?
    - Very important part of project's, especially in a team environent, but maybe not in this project
    - Continous Integration was very helpful
- Slow moving project?
- Using Python
    - Best tool for Machine Learning
    - But was inexperienced with the langauge, might have added too much learning load
- Focus of the project, sometimes unclear
- Changing Sprint lenght was very useful

# Tuesday

- Providing background on NLP
- Adding Background Chapter Sections to Project Report Docuemnt, updating/adding  references for the report
- Reading Chapter 2 in Practical NLP Book - Notability Notes
    - Advanced Preprocessing
    - Feature Engineering
- Maybe pre-processing is not meant to improve accuracy but remove/reduce overfitting, improving general performance even if accuracy is not good
    - This is a subjective measure
    - Need something more concrete — cross validation

# Wednesday

- I assumed that the datasets that my supervisor would provide me were currencies because that is what we had discussed would be a good starting point, but it was Dow Jones Data, but I decided to focus on it first

## Review/Retrospective

- My motivation/productivity improved this Sprint, hope to continue that into the upcoming Sprint
- Project Demo went better than expected personally
- Go the Productive Application finished, is in a good state with tests, CI and more
- Report is starting to take shape

## Planning

- Cross-validation
- Count of positive and negative words
- Start collecting my own data
    - Should this data be labelled with sentiment positive, negative and neutral

- Created an epic for 'Improving Price for the DJIA' and 'Gaining Background  Knowledge and Baclground Chapter Write-up'
    - Moving all relevant stories into these Epics
    - Closing down the 'Initial Approach' epic for now
        - It has been around since the start and was vague, now i have a more concrete focus so it can be replaced by the newly created Epics
    - Added epic for producing technical work too
    - 

## Notes

- An estimation has been hard for most of the work because it is all new to me, so it is often wrong, but it is hopefully consistently wrong. Need to check historical data
- How to frame report related stories to a customer?
    - Framed it in the perspective of the supervisor & second marker (markers of the project)
- Reading Practical NLP earlier would have been good
    - From the start of the project, would have helped me
    - Would have given me a goo in-depth understanding of NLP and the process required
- Reading Chapter 2, Section on Evaluation
    - Lots of metrics are avaialbe, See notability notes
- Adding Confusion Matrix something to consider when improving performance/evaluation measures
- ML and NLP Topic completely new to me, maybe too new
- Include statistics in the evaluation, based off the Sprints
    - Story Points completed in total
        - Useful work?
- Sometimes did not know how to organise the issues
    - Ended up adding the Epics for the reports — did not fit customer end goal befreo, but I framed them in the sense of the markers
    - Was hard to identify the customer/user in this scenario
- Touch on COVID, coronavirus in Evaluation
    - Affected my focus, motivation
    - Not even at the Univeristy, at Home in Norther Ireland. Worked better when in Aberystwyth