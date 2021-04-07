import os
import datetime

import pandas as pd

MONTH_NUMS = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

ORG_HEADLINE_CSV = './tesla_headlines.csv'
UPDATED_HEADLINE_CSV = './tesla_headlines_updated.csv'

# update headlines dates to ISO format
headlines_df = pd.read_csv(ORG_HEADLINE_CSV)

for row in range(0, len(headlines_df.index)):
    date = str(headlines_df.iloc[row, 0])
    print(date)
    split_date = date.split()
    month = MONTH_NUMS[split_date[0]]
    day = split_date[1]
    day = int(day[:-1])
    year = int(split_date[2])
    updated_date = datetime.date(year=year, month=month, day=day).isoformat()
    print(updated_date)
    headlines_df.iloc[row, 0] = updated_date

if os.path.exists(UPDATED_HEADLINE_CSV):
  os.remove(UPDATED_HEADLINE_CSV)
headlines_df.to_csv(path_or_buf=UPDATED_HEADLINE_CSV)


