
# coding: utf-8

# In[1]:

# Setup (Imports)
from datetime import datetime, timedelta
from collections import defaultdict

import requests
import random
import os
import re


# In[2]:


def strip_headline(headline):
    """Clean headline"""
    headline = headline.lower()
    headline = re.sub(r'^https?:\/\/.*[\r\n]*', '', headline, flags=re.MULTILINE)
    headline = ''.join(c for c in headline if c not in ",.?!;'\"{}[]()*#&:\\/|")
    return headline.strip()


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

def get_reuters_news(stock, limit=600):
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

def get_twitter_news(querys, limit=100):
    """Get headlines from Twitter"""
    from twitter import Twitter, OAuth
    import twitter_creds as c # Self-Created Python file with Creds

    twitter = Twitter(auth=OAuth(c.ACCESS_TOKEN, c.ACCESS_SECRET, c.CONSUMER_KEY, c.CONSUMER_SECRET))
    
    limit = min(limit, 100)
    
    articles = defaultdict(list)
    
    for query in querys:
    
        tweets = twitter.search.tweets(q=query, result_type='popular', lang='en', count=limit)['statuses']
        
        for tweet in tweets:
            
            text = re.sub(r'\W+', '', tweet['text'])
            date = tweet['created_at']
            
            if '\n' not in text and len(text) > len(query):
                
                date_key = datetime.strptime(date, "%a %b %d %H:%M:%S %z %Y" ).strftime('%Y-%m-%d')
                
                articles[date_key].append(text)
                
    return articles


# In[4]:


def save_headlines(headlines):
    """Save headlines to file"""
    with open(os.path.join('..', 'data', "_".join(headlines.keys()) + '-headlines.csv'), 'w', encoding="utf-8") as headline_file:
        
        for stock in headlines:
            
            articles = defaultdict(dict)

            for source, source_headlines in headlines[stock].items():

                for date in source_headlines:
                    
                    articles[date][source] = strip_headline(random.choice(headlines[stock][source][date]))
        
            for date in sorted(articles):

                current_articles = articles[date]

                headline_file.write("{},{},{}\n".format(stock, date, str(current_articles)))


# In[5]:


if __name__ == "__main__":
    
    headlines = {
            'GOOG': {
                'reddit': get_reddit_news(['google', 'Android', 'GooglePixel', 'news'], ['Google', 'pixel', 'android']), 
                'reuters': get_reuters_news('GOOG.O'),
                'twitter': get_twitter_news(['#Google', '#googlepixel', '#Alphabet'])
            },
            'AAPL': {
                'reddit': get_reddit_news(['apple', 'ios', 'AAPL', 'news'], ['apple', 'iphone', 'ipad', 'ios']), 
                'reuters': get_reuters_news('AAPL.O'),
                'twitter': get_twitter_news(['#Apple', '#IPhone', '#ios'])
            }
    }


# In[6]:


if __name__ == "__main__":

    save_headlines(headlines)

