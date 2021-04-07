import os
import pandas as pd

UPDATED_HEADLINE_CSV = './tesla_headlines_updated.csv'
LABELS_CSV = './tesla_price_labels.csv'
MERGED_CSV = './tesla_merged.csv'

headlines_df = pd.read_csv(UPDATED_HEADLINE_CSV)
merged_df = pd.read_csv(LABELS_CSV)

merged_dict = {}
merged_list = []

for labels_row in range(0, len(merged_df.index)):
    for headlines_row in range(0, len(headlines_df.index)):
        labels_date = str(merged_df.iloc[labels_row, 0])
        headlines_date = str(headlines_df.iloc[headlines_row, 1])

        if labels_date == headlines_date:
            merged_list.append(merged_df.iloc[labels_row, 1])
            for headlines_col in range(2, len(headlines_df.columns)):
                headline = str(headlines_df.iloc[headlines_row, headlines_col])
                if headline != 'nan':
                    print(headline)
                    merged_list.append(headline)
            merged_dict[labels_date] = merged_list
            merged_list = []

merged_df = pd.DataFrame.from_dict(merged_dict, orient='index')
if os.path.exists(MERGED_CSV):
  os.remove(MERGED_CSV)
merged_df.to_csv(path_or_buf=MERGED_CSV)
