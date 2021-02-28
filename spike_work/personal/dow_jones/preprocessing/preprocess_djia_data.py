"""
This script pre-processes the Dow Jones Industrial Average
headlines dataset, and exports the 'cleaned'/processed data
to another CSV file 
"""

import pandas as pd

import preprocess

DJIA_DATASET_PATH = '../../datasets/existing/dow_jones/Combined_News_DJIA.csv'
NUM_COLS = 25
DJIA_PROCESSED_DATASET_PATH = './djia_processed.csv'

df = pd.read_csv(DJIA_DATASET_PATH, nrows=10)

for i in range(1, NUM_COLS + 1):
    col = 'Top' + str(i)
    print(f'Column {col} processed...')
    df[col] = df[col].apply(lambda text: preprocess.apply_all(text))

df.to_csv(path_or_buf=DJIA_PROCESSED_DATASET_PATH, index=False)
