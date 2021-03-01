import pandas as pd

def combine_headline_rows(data_frame):
    combined_headlines = []

    for row in range(0, len(data_frame.index)):
        combined_headlines_in_row = ''
        for headline in data_frame.iloc[row, 2:27]:
            headline_str = str(headline)
            combined_headlines_in_row = combined_headlines_in_row + ' ' + headline_str
        # remove dangling spaces at beginning and end, trim?
        combined_headlines.append(combined_headlines_in_row.strip())

    return combined_headlines
