
# coding: utf-8

# In[1]:

# Setup (Imports)
from datetime import datetime, timedelta
from collections import defaultdict

import requests
import random
import os
import re

import yqd

from Database import add_stock_ticks, add_headlines, clean_ticks, db


# In[2]:


def consume_ticker_csv(stock, filename):
    """Loads data from csv file into database"""
    entries = []
    
    with open(os.path.join('..', 'data', filename), 'r') as tick_csv:
        
        for line in tick_csv:
            
            if "Date" not in line:
                
                date, open_, high, low, close, adj_close, volume = line.split(',')
                
                entries.append((stock, date, open_, high, low, close, adj_close, volume))
                
    add_stock_ticks(entries)
    
    clean_ticks()

def dl_ticker(stock, num_days=10):
    """Loads data from yahoo"""
    entries = []
    
    end_date = datetime.today()
    begin_date = end_date - timedelta(days=num_days)
    
    for line in yqd.load_yahoo_quote(stock, begin_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')):
        
        if "Date" not in line and len(line) > 1:
                
                date, open_, high, low, close, adj_close, volume = line.split(',')
                
                entries.append((stock, date, open_, high, low, close, adj_close, volume))
                
    add_stock_ticks(entries)
    
    clean_ticks()
    


# In[3]:


def basic_clean(text):
    return text.strip().replace("&#39;", "'").replace("&quot;", "").replace("&amp;", "and").replace("(TM)", "")

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

def get_reuters_news(stock, pages=80):
    """Get headlines from Reuters"""
    print('Downloading Reuters: ' + stock)
    
    found_headlines = []
    
    articles = defaultdict(list)
    
    pattern_headline = re.compile('<h2><a [\s\S]+?>([\s\S]+?)<\/a>[\s\S]*?<\/h2>')
    
    date_current = datetime.now()
    
    while pages > 0:

        text = requests.get('http://www.reuters.com/finance/stocks/company-news/{}?date={}'.format(stock, date_current.strftime('%m%d%Y')),  headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'}).text
        
        for match in pattern_headline.finditer(text):
            
            headline = match.group(1)
            
            headline = headline.replace('\u200d', '').replace('\u200b', '')
            
            headline = re.sub('^[A-Z]+[A-Z\d\s]*\-', '', headline)
            
            date_key = date_current.strftime('%Y-%m-%d')
            
            if headline not in found_headlines:
            
                articles[date_key].append(headline)
                found_headlines.append(headline)
        
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

def get_seekingalpha_news(stock, pages=500):
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
            
        try:

            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'}, cookies=cookies)
        
        except Exception as e:
            
            print(e)
            continue
    
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

def get_fool_news(stock, pages=40):
    "Get headlines from Motley Fool"
    print('Downloading MotleyFool: ' + stock)
    
    stock = stock.lower()
    
    re_headline = re.compile('<article id="article-\d+">[\s\S]+?">([\s\S]+?)<\/a>[\s\S]+?calendar"><\/i>([\s\S]+?20\d{2})')
    
    articles = defaultdict(list)
    
    for i in range(pages):
        
        if i == 0:
            url = "https://www.fool.com/quote/nasdaq/apple/{}/content".format(stock)
        else:
            url = "https://www.fool.com/quote/nasdaq/apple/{}/content/more?page={}".format(stock, i)
            
        try:
            
            text = requests.get(url).text
            
        except: # Timeout or something...
            
            pass
        
        headlines = [(match.group(1), match.group(2)) for match in re_headline.finditer(text)]
        
        for headline, date in headlines:
            
            date = datetime.strptime(date.strip(), "%b %d %Y")
            headline = basic_clean(headline)
            
            articles[date.strftime('%Y-%m-%d')].append(headline)
            
    return articles

def get_wsj(stock, pages=20):
    "Get headlines from WSJ"
    print('Downloading WSJ: ' + stock)
    
    re_headline = re.compile('<li class="\s+cr_pressRelease">[\s\S]+?"cr_dateStamp">([\s\S]+?)<\/li>[\s\S]+?href="http:\/\/www.wsj.com\/articles\S+?>([\s\S]+?)<\/a>')
    re_nextlink = re.compile('article_datetime" value="?(\d+\/\d+\/\d+)"?>[\s\S]+?article_docId" value="?(\d+)"?>[\s\S]+?newswire_datetime" value="?(\d+\/\d+\/\d+)"?>[\s\S]+?newswire_docId" value="?(\d+)"?>[\s\S]+?')
    
    url = "http://quotes.wsj.com/ajax/overview/5/US/{}?instrumentType=STOCK&significant=false".format(stock)
    
    articles = defaultdict(list)
    
    for i in range(pages):
        
        text = requests.get(url).text
        
        headlines = [(match.group(1), match.group(2)) for match in re_headline.finditer(text)]
        
        for date, headline in headlines:
            
            try:
                date = datetime.strptime(date.strip(), "%m/%d/%y")
            except:
                date = datetime.today()
                
            headline = basic_clean(headline)
            
            articles[date.strftime('%Y-%m-%d')].append(headline)
            
        nextpage_creds = re_nextlink.search(text)
        
        if nextpage_creds:
            
            nextpage_creds = [nextpage_creds.group(1), nextpage_creds.group(2), nextpage_creds.group(3), nextpage_creds.group(4)]
            url = "http://quotes.wsj.com/ajax/overview/5/US/{}?instrumentType=STOCK&significant=false&article_datetime={}&article_docId={}&newswire_datetime={}&newswire_docId={}".format(stock, *nextpage_creds)
        
        else:
            
            break
            
    return articles

def get_thestreet(stock, pages=60):
    "Get headlines from TheStreet"
    print('Downloading TheStreet: ' + stock)
    
    url = "https://www.thestreet.com/quote/{}/details/news?start=0&type=json".format(stock)
    
    articles = defaultdict(list)
    
    for i in range(pages):
    
        try:
            json = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'}).json()
        except:
            break
        
        for story in json['stories']:
            
            if 'headline' and 'callout' in story and story['headline'] and story['callout']:
            
                headline = basic_clean(story['headline'])
                callout = basic_clean(story['callout'])
                date = datetime.strptime(story['publishDate'], "%Y-%m-%dT%H:%M:%SZ")

                articles[date.strftime('%Y-%m-%d')].append(headline)
                articles[date.strftime('%Y-%m-%d')].append(callout)
            
        url = "https://www.thestreet.com" + json['pagination']['nextDataUrl']
        
    return articles


# In[4]:


def clean_headline(headline, dictionary):
    """
    Clean headline
    
    Removes extra chars and replaces words
    """
    headline = headline.lower()
    headline = re.sub('\d+%', 'STAT', headline)
    headline = re.sub('\b\d+\b', 'STAT', headline)
    headline = ''.join(c for c in headline if c in "abcdefghijklmnopqrstuvwxyz ")
    headline = re.sub('\s+', ' ', headline)
        
    for (word, replacement) in dictionary:
            
        headline = headline.replace(word, replacement)
        
    headline = headline.replace('STAT', '**STATISTIC**')
        
    headline = headline.replace('****', '** **') # Seperate joined kwords
    
    return headline.strip()


# In[5]:


def save_headlines(headlines):
    """Save headlines to file"""
    
    for stock in headlines:
        
        entries = []
        
        with db() as (conn, cur):
        
            cur.execute("SELECT word, replacement FROM dictionary WHERE stock=? ORDER BY LENGTH(word) DESC", [stock])
            dictionary = cur.fetchall()
        
        for source in headlines[stock]:
            
            for date in headlines[stock][source]:
                
                for headline in headlines[stock][source][date]:
                    
                    cleaned_headline = clean_headline(headline, dictionary)
                    
                    entries.append((stock, date, source, cleaned_headline, headline, -999))
                    
        add_headlines(entries)
    


# In[6]:


if __name__ == "__main__":
    
    headlines = {
            'GOOG': {
                'reddit': get_reddit_news(['google', 'Android', 'GooglePixel', 'news'], ['Google', 'pixel', 'android', 'stock']), 
                'reuters': get_reuters_news('GOOG.O'),
                'twitter': get_twitter_news(['@Google', '#Google', '#googlepixel', '#Alphabet']),
                'seekingalpha': get_seekingalpha_news('GOOG'),
                'fool': get_fool_news('GOOG'),
                'wsj': get_wsj('GOOG'),
                'thestreet': get_thestreet('GOOG')
            },
            'AAPL': {
                'reddit': get_reddit_news(['apple', 'ios', 'AAPL', 'news'], ['apple', 'iphone', 'ipad', 'ios', 'stock']), 
                'reuters': get_reuters_news('AAPL.O'),
                'twitter': get_twitter_news(['@Apple', '#Apple', '#IPhone', '#ios']),
                'seekingalpha': get_seekingalpha_news('AAPL'),
                'fool': get_fool_news('AAPL'),
                'wsj': get_wsj('AAPL'),
                'thestreet': get_thestreet('AAPL')
            },
            'MSFT': {
                'reddit': get_reddit_news(['microsoft', 'windowsphone', 'windows'], ['microsoft', 'phone', 'windows', 'stock']), 
                'reuters': get_reuters_news('MSFT.O'),
                'twitter': get_twitter_news(['@Microsoft', '#Windows', '#Microsoft', '#windowsphone']),
                'seekingalpha': get_seekingalpha_news('MSFT'),
                'fool': get_fool_news('MSFT'),
                'wsj': get_wsj('MSFT'),
                'thestreet': get_thestreet('MSFT')
            },
            'AMD': {
                'reddit': get_reddit_news(['Amd', 'AMD_Stock', 'pcmasterrace'], ['AMD', 'radeon', 'ryzen', 'stock']), 
                'reuters': get_reuters_news('AMD.O'),
                'twitter': get_twitter_news(['@AMD', '#AMD', '#Ryzen', '#radeon']),
                'seekingalpha': get_seekingalpha_news('AMD'),
                'fool': get_fool_news('AMD'),
                'wsj': get_wsj('AMD'),
                'thestreet': get_thestreet('AMD')
            },
            'AMZN': {
                'reddit': get_reddit_news(['amazon', 'amazonprime', 'amazonecho'], ['amazon', 'echo', 'prime', 'stock']), 
                'reuters': get_reuters_news('AMZN.O'),
                'twitter': get_twitter_news(['@amazon', '#Amazon', '#jeffbezos', '@amazonecho', '#amazonprime']),
                'seekingalpha': get_seekingalpha_news('AMZN'),
                'fool': get_fool_news('AMZN'),
                'wsj': get_wsj('AMZN'),
                'thestreet': get_thestreet('AMZN')
            },
            'INTC': {
                'reddit': get_reddit_news(['intel', 'hardware'], ['intel', 'cpu']),
                'reuters': get_reuters_news('INTC.O'),
                'twitter': get_twitter_news(['@intel']),
                'seekingalpha': get_seekingalpha_news('INTC'),
                'fool': get_fool_news('INTC'),
                'wsj': get_wsj('INTC'),
                'thestreet': get_thestreet('INTC')
            }
    }


# In[7]:


if __name__ == "__main__":

    save_headlines(headlines)


# In[8]:


if __name__ == "__main__":
    
    dl_ticker('AAPL')
    dl_ticker('AMZN')
    dl_ticker('AMD')
    dl_ticker('GOOG')
    dl_ticker('MSFT')
    dl_ticker('INTC')

