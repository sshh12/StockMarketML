
# coding: utf-8

# In[ ]:

# Setup (Imports)

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from gensim.models import Word2Vec

from collections import defaultdict
from datetime import datetime
from praw import Reddit

import requests
import os


# In[ ]:


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


# In[ ]:


reddit = Reddit('StockMarketML')

articles = defaultdict(list)
sentences = []

for submission in reddit.subreddit('news+apple+ios+AAPL').search('apple', limit=None):
    
    articles[datetime.fromtimestamp(submission.created).strftime('%Y-%m-%d')].append(submission.title)
    
    sentences.append(submission.title)
    
print(convert_sentences_to_vector(sentences)[0])
    
with open(os.path.join('data', 'reddit.csv'), 'w') as redditfile:
    
    for date, sents in articles.items():
        
        data = str(sents).encode("utf-8")
    
        redditfile.write(date + ", " + str(data)[1:] + "\n")
    

