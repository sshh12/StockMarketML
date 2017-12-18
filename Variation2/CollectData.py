
# coding: utf-8

# In[1]:

# Setup (Imports)

from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

from datetime import datetime, timedelta
from collections import defaultdict

import requests
import random
import os
import re


# In[2]:


def strip_headline(headline):
    """Clean headline"""
    return headline.replace(',', '')


# In[3]:


def get_reddit_news(subs, search_terms, limit=None, praw_config='StockMarketML'):
    "Get headlines from Reddit"
    from praw import Reddit
    
    reddit = Reddit(praw_config)

    articles = defaultdict(list)
    
    used = []
    
    for term in search_terms:

        for submission in reddit.subreddit('+'.join(subs)).search(term, limit=limit):
            
            if submission.title not in used:
                
                used.append(submission.title)
                
                date_key = datetime.fromtimestamp(submission.created).strftime('%Y-%m-%d')

                articles[date_key].append(submission.title)
        
    return articles

def get_reuters_news(stock, limit=400):
    "Get headlines from Reuters"
    articles = defaultdict(list)
    
    pattern_headline = re.compile('<h2>\s*(<a [\S]*\s*>)?(.+?)(<\/a>)?\s*<\/h2>')
    
    date_current = datetime.now()
    
    while limit > 0:
        
        text = requests.get('http://www.reuters.com/finance/stocks/company-news/{}?date={}'.format(stock, date_current.strftime('%m%d%Y'))).text
        
        for match in pattern_headline.finditer(text):
            
            headline = match.group(2)
            
            headline = re.sub('[A-Z][A-Z\d\s]{5,}\-', '', headline)
            
            date_key = date_current.strftime('%Y-%m-%d')
            
            if headline not in articles[date_key]:
            
                articles[date_key].append(headline)
        
            limit -= 1
        
        date_current -= timedelta(days=1)
        
    return articles

def save_headlines(headlines, force_one_per_day=False):
    """Save headlines to file"""
    with open(os.path.join('..', 'data', "_".join(headlines.keys()) + '-headlines.csv'), 'w', encoding="utf-8") as headline_file:
        
        for stock in headlines:
    
            articles = defaultdict(list)

            for source in headlines[stock]:

                for date in source:

                    articles[date].extend(source[date])
        
            for date in sorted(articles):

                current_articles = articles[date]

                if force_one_per_day:

                    current_articles = [random.choice(current_articles)]

                for headline in current_articles:

                    headline_file.write("{},{},{}\n".format(stock, date, strip_headline(headline)))


# In[4]:


if __name__ == "__main__":
    
    headlines = {
        'AAPL': [
            get_reddit_news(['apple', 'ios', 'AAPL', 'news'], ['apple', 'iphone', 'ipad', 'ios'], limit=10), 
            get_reuters_news('AAPL.O', limit=10)
        ]
    }


# In[5]:


if __name__ == "__main__":

    save_headlines(headlines, force_one_per_day=True)

