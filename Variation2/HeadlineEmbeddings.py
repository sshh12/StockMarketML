
# coding: utf-8

# In[ ]:

# Imports

import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


def get_tick_data(stocks):
    
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


# In[ ]:


def get_headline_data(stocks):
    
    history = {}
    
    with open(os.path.join('..', 'data', "_".join(stocks) + '-headlines.csv'), 'r') as data:
        
        for line in data:

            if len(line) > 6:

                stock, date, headline = line.split(",")
                
                if not stock in history:
                    
                    history[stock] = {}
                
                history[stock][date] = headline.replace('\n', '')
                
    return history

