
# coding: utf-8

# In[15]:

# Setup (Imports)
from datetime import datetime, timedelta
from collections import defaultdict

import requests
import random
import os
import re


# In[16]:


def clean_headline(headline, replacements={}):
    """
    Clean headline
    
    Removes extra chars and replaces words
    """
    headline = headline.lower()
    headline = ''.join(c for c in headline if c not in ",.?!;-'\‘’\"{}[]()*#&:\\/@|0123456789$%")
    headline = re.sub('\s+', ' ', headline)
    
    for original, replacement in replacements.items():
        headline = headline.replace(original, replacement)
    
    return headline.strip()


# In[17]:


def get_reddit_news(subs, search_terms, limit=None, praw_config='StockMarketML'):
    "Get headlines from Reddit"
    print('Downloading Reddit Posts: ' + ", ".join(subs))
    
    from praw import Reddit
    
    reddit = Reddit(praw_config)

    articles = defaultdict(list)
    
    used = []
    
    for term in search_terms:

        for submission in reddit.subreddit('+'.join(subs)).search(term, limit=limit):
            
            if submission.title.count(' ') > 4 and submission.title not in used:
                
                used.append(submission.title)
                
                date_key = datetime.fromtimestamp(submission.created).strftime('%Y-%m-%d')

                articles[date_key].append(submission.title)
        
    return articles

def get_reuters_news(stock, pages=100):
    """Get headlines from Reuters"""
    print('Downloading Reuters: ' + stock)
    
    articles = defaultdict(list)
    
    pattern_headline = re.compile('<h2><a [\s\S]+>([\s\S]+)<\/a>[\s\S]*<\/h2>')
    
    date_current = datetime.now()
    
    while pages > 0:
        
        text = requests.get('http://www.reuters.com/finance/stocks/company-news/{}?date={}'.format(stock, date_current.strftime('%m%d%Y')), allow_redirects=False, timeout=3, headers={'User-Agent': 'StockMarketBot'}).text
        
        for match in pattern_headline.finditer(text):
            
            headline = match.group(1)
            
            headline = re.sub('[A-Z][A-Z\d\s]{5,}\-', '', headline)
            
            date_key = date_current.strftime('%Y-%m-%d')
            
            if headline not in articles[date_key]:
            
                articles[date_key].append(headline)
        
        pages -= 1
        
        date_current -= timedelta(days=1)
        
    return articles

def get_twitter_news(querys, limit=100):
    """Get headlines from Twitter"""
    print('Downloading Tweets: ' + ", ".join(querys))
    
    from twitter import Twitter, OAuth
    import twitter_creds as c # Self-Created Python file with Creds

    twitter = Twitter(auth=OAuth(c.ACCESS_TOKEN, c.ACCESS_SECRET, c.CONSUMER_KEY, c.CONSUMER_SECRET))
    
    limit = min(limit, 100)
    
    articles = defaultdict(list)
    
    for query in querys:
    
        tweets = twitter.search.tweets(q=query, result_type='popular', lang='en', count=limit)['statuses']
        
        for tweet in tweets:
            
            text = re.sub(r'https?:\/\/\S+', '', tweet['text'])
            text = re.sub(r'[^\w\s:/]+', '', text)
            
            date = tweet['created_at']
            
            if '\n' not in text and len(text) > len(query) and ' ' in text:
                
                date_key = datetime.strptime(date, "%a %b %d %H:%M:%S %z %Y" ).strftime('%Y-%m-%d')
                
                articles[date_key].append(text)
                
    return articles

def get_seekingalpha_news(stock, pages=300):
    """Get headlines from SeekingAlpha"""
    print('Downloading SeekingAlpha: ' + stock)

    articles = defaultdict(list)

    re_headline = re.compile('<a class="market_current_title" [\s\S]+?>([\s\S]+?)<\/a>')
    re_dates = re.compile('<span class="date pad_on_summaries">([\s\S]+?)<\/span>')

    cookies = None

    for i in range(1, pages + 1):

        if i == 1:
            url = 'https://seekingalpha.com/symbol/{}/news'.format(stock)
        else:
            url = 'https://seekingalpha.com/symbol/{}/news/more_news_all?page={}'.format(stock, i)

        r = requests.get(url, headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'}, cookies=cookies)

        text = r.text.replace('\\"', '"')
        cookies = r.cookies # SeekingAlpha wants cookies.

        headlines = [match.group(1) for match in re_headline.finditer(text)]
        dates = [match.group(1) for match in re_dates.finditer(text)]

        for headline, date in zip(headlines, dates):
            
            headline = headline.replace('(update)', '')
            
            date = date.replace('.', '')

            if 'Today' in date:
                date = datetime.today()
            elif 'Yesterday' in date:
                date = datetime.today() - timedelta(days=1)
            else:
                temp = date.split(',')
                if len(temp[0]) == 3:
                    date = datetime.strptime(temp[1], " %b %d").replace(year=datetime.today().year)
                else:
                    date = datetime.strptime("".join(temp[0:2]), "%b %d %Y")

            articles[date.strftime('%Y-%m-%d')].append(headline)

    return articles


# In[18]:


def save_headlines(headlines, kword_replacements={}):
    """Save headlines to file"""
    with open(os.path.join('..', 'data', "_".join(sorted(headlines.keys())) + '-headlines.csv'), 'w', encoding="utf-8") as headline_file:
        
        for stock in headlines:
            
            # Converting Stock -> Source -> Date -> Headlines
            #         to Stock -> Date -> Source -> Headline
            
            articles = defaultdict(dict)

            for source, source_headlines in headlines[stock].items():

                for date in source_headlines:
                    
                    articles[date][source] = clean_headline(random.choice(headlines[stock][source][date]), kword_replacements[stock])
        
            for date in sorted(articles):

                current_articles = articles[date]

                headline_file.write("{},{},{}\n".format(stock, date, str(current_articles).replace(',', '@')))


# In[19]:


if __name__ == "__main__":
    
    headlines = {
            'GOOG': {
                'reddit': get_reddit_news(['google', 'Android', 'GooglePixel', 'news'], ['Google', 'pixel', 'android', 'stock']), 
                'reuters': get_reuters_news('GOOG.O'),
                'twitter': get_twitter_news(['@Google', '#Google', '#googlepixel', '#Alphabet']),
                'seekingalpha': get_seekingalpha_news('GOOG')
            },
            'AAPL': {
                'reddit': get_reddit_news(['apple', 'ios', 'AAPL', 'news'], ['apple', 'iphone', 'ipad', 'ios', 'stock']), 
                'reuters': get_reuters_news('AAPL.O'),
                'twitter': get_twitter_news(['@Apple', '#Apple', '#IPhone', '#ios']),
                'seekingalpha': get_seekingalpha_news('AAPL')
            },
            'MSFT': {
                'reddit': get_reddit_news(['microsoft', 'windowsphone', 'windows'], ['microsoft', 'phone', 'windows', 'stock']), 
                'reuters': get_reuters_news('MSFT.O'),
                'twitter': get_twitter_news(['@Microsoft', '#Windows', '#Microsoft', '#windowsphone']),
                'seekingalpha': get_seekingalpha_news('MSFT')
            },
            'AMD': {
                'reddit': get_reddit_news(['Amd', 'AMD_Stock', 'pcmasterrace'], ['AMD', 'radeon', 'ryzen', 'stock']), 
                'reuters': get_reuters_news('AMD.O'),
                'twitter': get_twitter_news(['@AMD', '#AMD', '#Ryzen', '#radeon']),
                'seekingalpha': get_seekingalpha_news('AMD')
            }
    }


# In[20]:


if __name__ == "__main__":
    
    kword_replacements = { # To generalize headlines
        'GOOG': {
            'google': '**COMPANY**',
            'alphabet': '**COMPANY**',
            'android': '**PRODUCT**',
            'pixel': '**PRODUCT**'
        },
        'AAPL': {
            'apple': '**COMPANY**', 
            'macbook': '**PRODUCT**',
            'iphone': '**PRODUCT**',
            'ipad': '**PRODUCT**'
            'ios': '**PRODUCT**'
        },
        'MSFT': {
            'microsoft': '**COMPANY**',
            'windows': '**PRODUCT**'
        },
        'AMD': {
            'amd': '**COMPANY**', 
            'ryzen': '**PRODUCT**',
            'radeon': '**PRODUCT**'
        },
    }

    save_headlines(headlines, kword_replacements)

