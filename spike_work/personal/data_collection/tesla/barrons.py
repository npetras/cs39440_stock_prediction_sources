"""
https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638
"""
from urllib.request import urlopen, Request
import requests
from bs4 import BeautifulSoup


url = 'https://www.barrons.com/quote/stock/us/xnas/tsla'
req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
response = urlopen(req)    
# Read the contents of the file into 'html'
html = BeautifulSoup(response)
# Find 'news-table' in the Soup and load it into 'news_table'
news_list = html.find(class_='news-columns')
list_items = news_list.findAll('li')

for item in list_items:
    date = item.span.text
    headline = item.a.text
    print(date)
    print(headline
