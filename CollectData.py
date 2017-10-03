
# coding: utf-8

# In[1]:

# Setup (Imports)

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from gensim.models import Word2Vec

from datetime import datetime, timedelta
from collections import defaultdict

import requests
import os
import re


# In[2]:


def process_raw_text(text):

    tokenizer = RegexpTokenizer(r'\w+')
    text_processed = tokenizer.tokenize(text)
    
    text_processed = [word.lower() for word in text_processed if word.lower() not in stopwords.words('english')]

    porter_stemmer = PorterStemmer()

    text_processed = [porter_stemmer.stem(word) for word in text_processed]

    return " ".join(text_processed)

def convert_sentences_to_vector(sentences):
    
    sentences = list(map(process_raw_text, sentences))
    
    dictionary = []
    
    for sentence in sentences:
        
        dictionary.append(sentence.split(' '))
        
    word_model = Word2Vec(dictionary, size=100, window=5, min_count=1, workers=4)
    word_model.save(os.path.join('models', 'word2vec.model'))
    
    vector = [[word_model.wv[word] for word in sentence.split(' ')] for sentence in sentences]
    
    return vector


# In[3]:


def get_reddit_news(subs, search_term, limit=None, praw_config='StockMarketML'):
    
    from praw import Reddit
    
    reddit = Reddit(praw_config)

    articles = defaultdict(list)

    for submission in reddit.subreddit('+'.join(subs)).search(search_term, limit=limit):
    
        articles[datetime.fromtimestamp(submission.created).strftime('%m/%d/%Y')].append(submission.title)
        
    return articles

def get_reuters_news(stock, limit=200):
    
    articles = defaultdict(list)
    
    pattern_headline = re.compile('<h2>\s*(<a [\S]*\s*>)?(.+?)(<\/a>)?\s*<\/h2>')
    
    date_current = datetime.now()
    
    while limit > 0:
        
        text = requests.get('http://www.reuters.com/finance/stocks/company-news/{}?date={}'.format(stock, date_current.strftime('%m%d%Y'))).text
        
        for match in pattern_headline.finditer(text):
            
            headline = match.group(2)
            
            articles[date_current.strftime('%m/%d/%Y')].append(headline)
        
            limit -= 1
        
        date_current -= timedelta(days=1)
        
    return articles

def get_yahoo_finance_news(suburl="", limit=1): # TODO FIX
    
    pattern_headline = re.compile('<u class="StretchedBox" data-reactid="\d+"><\/u><!-- react-text: \d+ -->(.+?)<!-- \/react-text --><\/a><\/h3>')
    
    url = "https://finance.yahoo.com/" + suburl 
    
    while limit > 0:
        
        text = requests.get(url).text
        
        for match in pattern_headline.finditer(text):
            
            headline = match.group(1)
            
            print(headline)
        
        limit -= 1


# In[4]:


# get_reddit_news(['news', 'apple', 'ios', 'AAPL'], 'apple')
# get_reuters_news('AAPL.O')
# get_yahoo_finance_news('tech/apple')

