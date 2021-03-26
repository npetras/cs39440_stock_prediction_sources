"""
https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638
"""
from urllib.request import urlopen, Request
import requests
from bs4 import BeautifulSoup


url = 'https://finviz.com/quote.ashx?t=amzn'
req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
response = urlopen(req)    
# Read the contents of the file into 'html'
html = BeautifulSoup(response)
# Find 'news-table' in the Soup and load it into 'news_table'
news_table = html.find(id='news-table')
tr = news_table.findAll('tr')

for table_row in tr:
    a_text = table_row.a.text
    td_text = table_row.td.text
    print(a_text)
    print(td_text)
