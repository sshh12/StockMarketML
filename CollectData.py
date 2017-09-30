
# coding: utf-8

# In[1]:

# Setup (Imports)

from collections import defaultdict

from datetime import datetime
from praw import Reddit
import requests


# In[2]:

reddit = Reddit('StockMarketML')

articles = defaultdict(list)

for submission in reddit.subreddit('news+apple+ios+AAPL').search('apple', limit=None):
    articles[datetime.fromtimestamp(submission.created).strftime('%Y-%m-%d')].append(submission.title)
    
articles

