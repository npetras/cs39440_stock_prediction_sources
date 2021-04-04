import pandas as pd
import os

price_df = pd.read_csv('./tesla_price.csv')
LABEL_CSV = './tesla_price_labels.csv'
label = None

labels_dict = {}

for row in range(0, len(price_df.index)):
    date = price_df.iloc[row, 0]
    open_price = price_df.iloc[row, 1]
    close_price = price_df.iloc[row, 4]


    # print(f'{date} {open_price} {close_price}')


    if((close_price - open_price) >= 0):
        labels_dict[date] = 1
        # print(1)
    else:
        labels_dict[date] = 00
        # print(0)

labels_df = pd.DataFrame.from_dict(labels_dict, orient='index', columns=['Label'])
print(labels_df.dtypes)
if os.path.exists(LABEL_CSV):
  os.remove(LABEL_CSV)
labels_df.to_csv(path_or_buf=LABEL_CSV)
