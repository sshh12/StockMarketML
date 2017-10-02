
# coding: utf-8

# In[1]:

# Setup (Imports)


from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from collections import defaultdict
from datetime import datetime
from praw import Reddit

import requests
import os


# In[2]:


def process_raw_text(text):

    tokenizer = RegexpTokenizer(r'\w+')
    text_processed = tokenizer.tokenize(text)
    
    text_processed = [word.lower() for word in text_processed if word.lower() not in stopwords.words('english')]

    porter_stemmer = PorterStemmer()

    text_processed = [porter_stemmer.stem(word) for word in text_processed]

    return " ".join(text_processed)


# In[3]:


reddit = Reddit('StockMarketML')

articles = defaultdict(list)

for submission in reddit.subreddit('news+apple+ios+AAPL').search('apple', limit=None):
    
    articles[datetime.fromtimestamp(submission.created).strftime('%Y-%m-%d')].append(submission.title)
    
with open(os.path.join('data', 'reddit.csv'), 'w') as redditfile:
    
    for date, sents in articles.items():
        
        data = str(sents).encode("utf-8")
    
        redditfile.write(date + ", " + str(data)[1:] + "\n")
    

