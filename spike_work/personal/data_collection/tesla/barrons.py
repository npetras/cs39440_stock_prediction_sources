"""
https://stackoverflow.com/questions/21006940/how-to-load-all-entries-in-an-infinite-scroll-at-once-to-parse-the-html-in-pytho
"""
import time
import re
import pprint
import os


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd

TIME_REGEX = '''\s\d:\d{2}\s[pa].m.\s.*'''
TESLA_CSV = './tesla_headlines.csv'
browser = webdriver.Firefox()

browser.get('https://www.barrons.com/quote/stock/tsla')
time.sleep(1)

scroll_elem = browser.find_element_by_id('barrons-news-infinite')

no_of_pagedowns = 20

while no_of_pagedowns:
    scroll_elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)
    no_of_pagedowns-=1

headlines = browser.find_elements_by_xpath('/html/body/section/div/div[4]/div[1]/div[1]/div/div[4]/div[2]/div/div/div/div[2]/div/div[1]/div/div/ul/li/a')
dates = browser.find_elements_by_xpath('/html/body/section/div/div[4]/div[1]/div[1]/div/div[4]/div[2]/div/div/div/div[2]/div/div[1]/div/div/ul/li/span[@class="date"]')

headlines_dict = {}
headlines_list = []
prev_date = ''

for i, headline in enumerate(headlines):
        date = dates[i].text
        date_wo_time = re.sub(TIME_REGEX, '', date)

        if prev_date == '':
            headlines_list.append(headline.text)
            prev_date = date_wo_time
        elif date_wo_time == prev_date:
            headlines_list.append(headline.text)
        elif date_wo_time is not prev_date:
            headlines_dict[prev_date] = headlines_list
            headlines_list = []
            headlines_list.append(headline.text)
            prev_date = date_wo_time


pp = pprint.PrettyPrinter()
pp.pprint(headlines_dict)


df = pd.DataFrame.from_dict(headlines_dict, orient='index')
print(df.dtypes)
if os.path.exists(TESLA_CSV):
  os.remove(TESLA_CSV)
df.to_csv(path_or_buf=TESLA_CSV)
