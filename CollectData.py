
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


def get_reddit_news(subs, search_terms, limit=None, praw_config='StockMarketML'):
    
    from praw import Reddit
    
    reddit = Reddit(praw_config)

    articles = defaultdict(list)
    
    used = []
    
    for term in search_terms:

        for submission in reddit.subreddit('+'.join(subs)).search(term, limit=limit):
            
            if submission.title not in used:
                
                used.append(submission.title)

                articles[datetime.fromtimestamp(submission.created).strftime('%Y-%m-%d')].append(submission.title)
        
    return articles

def get_reuters_news(stock, limit=400):
    
    articles = defaultdict(list)
    
    pattern_headline = re.compile('<h2>\s*(<a [\S]*\s*>)?(.+?)(<\/a>)?\s*<\/h2>')
    
    date_current = datetime.now()
    
    while limit > 0:
        
        text = requests.get('http://www.reuters.com/finance/stocks/company-news/{}?date={}'.format(stock, date_current.strftime('%m%d%Y'))).text
        
        for match in pattern_headline.finditer(text):
            
            headline = match.group(2)
            
            headline = re.sub('[A-Z][A-Z\d\s]{5,}\-', '', headline)
            
            articles[date_current.strftime('%Y-%m-%d')].append(headline)
        
            limit -= 1
        
        date_current -= timedelta(days=1)
        
    return articles

def save_headlines(stock, sources, force_one_per_day=False):
    
    articles = defaultdict(list)
    
    for source in sources:
        
        for date in source:
            
            articles[date].extend(source[date])
            
    with open(os.path.join('data', stock + '-headlines.csv'), 'w', encoding="utf-8") as headline_file:
        
        for date in sorted(articles):
            
            current_articles = articles[date]
            
            if force_one_per_day:
                
                current_articles = [random.choice(current_articles)]
        
            headline_file.write("{},{}\n".format(date, "@@".join(current_articles).replace(',', '')))


# In[3]:


if __name__ == "__main__":

    save_headlines('AAPL', 
                   [get_reddit_news(['apple', 'ios', 'AAPL', 'news'], ['apple', 'iphone', 'ipad', 'ios']), 
                    get_reuters_news('AAPL.O')], 
                   force_one_per_day=True)


# In[4]:


def process_raw_text(text):

    words = re.findall(r'\b\w+\b', text)
    
    cleaned = list(map(lambda w: w.lower(), words))

    return " ".join(cleaned)

def convert_headlines_to_vectors(stock, create_model=True):
    
    def read_headline_file():
        
        with open(os.path.join('data', stock + '-headlines.csv'), 'r', encoding="utf-8") as headline_file:
        
            for line in headline_file:
            
                if len(line) > 6:
                
                    date, headlines = line.split(',')
                
                    yield date, map(lambda x: x.strip(), headlines.split("@@"))
    
    if create_model:
        
        i = 0
        
        headlines_corpus = []
        
        for date, headlines in read_headline_file():
            
            for headline in headlines:
                
                if headline not in headlines_corpus:
                
                    headlines_corpus.append(LabeledSentence(process_raw_text(headline), tags=['headline_' + str(i)]))
                
                    i += 1
        
        doc_model = Doc2Vec(headlines_corpus, size=100, window=5, min_count=3, workers=4)
        doc_model.save(os.path.join('models', stock + '-headlines-doc2vec.model'))
    
    doc_model = Doc2Vec.load(os.path.join('models', stock + '-headlines-doc2vec.model'))
    
    with open(os.path.join('data', stock + '-headlines-vectors.csv'), 'w', encoding="utf-8") as headline_vectors:
        
        i = 0
        
        used = []
        
        for date, headlines in read_headline_file(): #TODO file read not needed
        
            for headline in headlines:
                
                if headline not in used:
                    
                    used.append(headline)
                
                    vector = doc_model.docvecs[i]
                
                    vector = str(list(vector))
                
                    headline_vectors.write("{},{}\n".format(date, vector))
                
                i += 1
    
    return doc_model


# In[5]:


if __name__ == "__main__":

    convert_headlines_to_vectors('AAPL')

