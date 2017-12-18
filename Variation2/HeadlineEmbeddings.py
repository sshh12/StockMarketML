
# coding: utf-8

# In[27]:

# Imports

from datetime import datetime, timedelta

import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[5]:


def get_tick_data(stocks):
    """
    Tick Data
    
    This reads the high, lows, closes, etc. from data csv files
    """
    history = {}
    
    for stock in stocks:
        
        history[stock] = {}
        
        with open(os.path.join('..', 'data', stock + '.csv'), 'r') as data:

            for line in data:

                if len(line) > 6 and "Date" not in line and "null" not in line:

                    items = line.split(",")
                    
                    date = items[0]
                    data = np.array(list(map(float, items[1:]))) # 0, 1, 2, 4, 5 -> OPEN HIGH LOW ADJ_CLOSE VOLUME
                    
                    history[stock][date] = data
        
    return history


# In[19]:


def get_headline_data(stocks):
    """
    Headline Data
    
    This reads the headlines from the headline csv file (created by CollectData)
    """
    history = {}
    
    with open(os.path.join('..', 'data', "_".join(stocks) + '-headlines.csv'), 'r') as data:
        
        for line in data:

            if len(line) > 6:

                stock, date, headline = line.split(",")
                
                if not stock in history:
                    
                    history[stock] = {}
                
                history[stock][date] = headline.replace('\n', '')
                
    return history


# In[38]:


def make_headline_to_effect_data(tick_data, head_data):
    """
    Headline -> Effect
    
    Creates essentially the X, Y data for the embedding model to use
    when analyzing/encoding headlines. Returns a list of headlines and
    a list of corresponding 'effects' which represent a change in the stock price.
    """
    headlines, effects = [], []
    
    for stock, dates in head_data.items():
        
        for date, headline in dates.items():
            
            next_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
            next_date = next_date.strftime('%Y-%m-%d')
            
            if date in tick_data[stock] and next_date in tick_data[stock]:
                
                tick_on = tick_data[stock][date]
                tick_after = tick_data[stock][next_date]
                
                if tick_after[3] < tick_on[3]:
                    
                    effects.append(-1)
                    
                else:
                    
                    effects.append(1)
                    
                headlines.append(headline)
                
    return headlines, effects


# In[39]:


def encode_sentences(sentences, max_length=10):
    """
    Encoder
    
    Takes a list of headlines and converts them into vectors
    """
    toke = Tokenizer()
    
    toke.fit_on_texts(sentences)
    
    vocab_size = len(toke.word_index) + 1
    
    encoded_headlines = toke.texts_to_sequences(sentences)
    
    padded_headlines = pad_sequences(encoded_headlines, maxlen=max_length, padding='post')
    
    return padded_headlines


# In[40]:


if __name__ == "__main__":
    
    stocks = ['AAPL']
    
    tick_data = get_tick_data(stocks)
    head_data = get_headline_data(stocks)
    
    headlines, effects = make_headline_to_effect_data(tick_data, head_data)
    
    encoded_headlines = encode_sentences(headlines)
    
    print(encoded_headlines)

