
# coding: utf-8

# In[1]:

# Imports

from datetime import datetime, timedelta

from Database import db
 
import numpy as np
import pickle
import os
import re

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

from keras.optimizers import RMSprop
from keras.models import Sequential, load_model, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, concatenate, SpatialDropout1D, GRU
from keras.layers import Dense, Flatten, Embedding, LSTM, Activation, BatchNormalization, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
import keras.backend as K
from keras.utils import plot_model


# In[2]:

# Options

stocks      = ['AAPL', 'AMD', 'AMZN', 'GOOG', 'MSFT', 'INTC']
all_sources = ['reddit', 'reuters', 'twitter', 'seekingalpha', 'fool', 'wsj', 'thestreet']

sample_size = 5
tick_window = 30
max_length  = 50
vocab_size  = None # Set by tokenizer
emb_size    = 300

epochs      = 120
batch_size  = 64


# In[3]:


def add_time(date, days):
    
    return (date + timedelta(days=days)).strftime('%Y-%m-%d')

def clean(sentence):
    
    sentence = sentence.lower()
    sentence = sentence.replace('-', ' ').replace('_', ' ').replace('&', ' ')
    sentence = re.sub('\$?\d+%?\w?', 'numbertoken', sentence)
    sentence = ''.join(c for c in sentence if c in "abcdefghijklmnopqrstuvwxyz ")
    sentence = re.sub('\s+', ' ', sentence)
    
    return sentence.strip()

def make_headline_to_effect_data():
    """
    Headline -> Effect
    
    Creates essentially the X, Y data for the embedding model to use
    when analyzing/encoding headlines. Returns a list of headlines and
    a list of corresponding 'effects' which represent a change in the stock price.
    """
    all_headlines, all_tick_hist, all_effects = [], [], []
    
    with db() as (conn, cur):
        
        for stock in stocks:
            
            ## Headline For Every Date ##
            
            cur.execute("SELECT DISTINCT date FROM headlines WHERE stock=? ORDER BY date ASC LIMIT 1", [stock])
            start_date = cur.fetchall()[0][0]
            
            cur.execute("SELECT DISTINCT date FROM ticks WHERE stock=? AND date >= ? ORDER BY date ASC", [stock, start_date])
            dates = [date[0] for date in cur.fetchall()]
            
            for date in tqdm_notebook(dates, desc=stock):
                
                ## Collect Headlines ##
                
                event_date = datetime.strptime(date, '%Y-%m-%d')
                
                cur.execute("SELECT date, source, rawcontent FROM headlines WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date DESC", 
                            [stock, add_time(event_date, -14), date])
                headlines = [(date, source, clean(content), (event_date - datetime.strptime(date, '%Y-%m-%d')).days) 
                                 for (date, source, content) in cur.fetchall() if content]
                
                if len(headlines) < sample_size:
                    continue
                    
                ## Find corresponding tick data ## 
                
                cur.execute("""SELECT open, high, low, adjclose, volume FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date DESC""", 
                            [stock, 
                             add_time(event_date, -30 - tick_window), 
                             add_time(event_date, 0)])
                
                before_headline_ticks = cur.fetchall()[:tick_window]
                
                if len(before_headline_ticks) != tick_window:
                    continue
                
                cur.execute("""SELECT AVG(adjclose) FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date""", 
                            [stock, 
                             add_time(event_date, 1), 
                             add_time(event_date, 4)])
                
                after_headline_ticks = cur.fetchall()
                
                if len(after_headline_ticks) == 0:
                    continue
                
                previous_tick = before_headline_ticks[0][3]
                result_tick = after_headline_ticks[0][0]
                
                if not previous_tick or not result_tick:
                    continue
                
                tick_hist = np.array(before_headline_ticks)
                tick_hist -= np.mean(tick_hist, axis=0)
                tick_hist /= np.std(tick_hist, axis=0)
                
                ## Create training example ##

                probs = [1 / (headline[3] + 1) for headline in headlines]
                probs /= np.sum(probs)
                    
                contents = [headline[2] for headline in headlines]

                num_samples = len(contents) // sample_size
                    
                effect = [(result_tick - previous_tick) / previous_tick]

                for i in range(num_samples):

                    indexes = np.random.choice(np.arange(len(headlines)), sample_size, replace=False, p=probs)
                    
                    sample = [headlines[i] for i in indexes]

                    all_headlines.append(sample)
                    all_tick_hist.append(tick_hist)
                    all_effects.append(effect)
                    
            break ## REMOVE BEFORE FLIGHT
                    
    return all_headlines, np.array(all_tick_hist), np.array(all_effects)


# In[4]:


def encode_sentences(headlines, tokenizer=None, max_length=100, vocab_size=100):
    """
    Encoder
    
    Takes a list of headlines and converts them into vectors
    """
    ## Encoding Sentences
    
    sentences = []
    
    for example in headlines:
        sentences.append(" ".join([data[2] for data in example]))
    
    if not tokenizer:
        
        tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False) # Already PreProcessed
    
        tokenizer.fit_on_texts(sentences)
    
    encoded_headlines = tokenizer.texts_to_sequences(sentences)
    
    padded_headlines = pad_sequences(encoded_headlines, maxlen=max_length, padding='post')
    
    ## Encoding Meta Data
    
    # TODO
    
    return padded_headlines, tokenizer


# In[5]:


def get_embedding_matrix(tokenizer, pretrained_file='glove.840B.300d.txt'):
    
    embedding_matrix = np.zeros((vocab_size + 1, emb_size))
    
    if not pretrained_file:
        return embedding_matrix, None
    
    ## Load Glove File (Super Slow) ##
    
    glove_db = dict()
    
    with open(os.path.join('..', 'data', pretrained_file), 'r', encoding="utf-8") as glove:

        for line in tqdm_notebook(glove, desc='Glove', total=2196017):

            values = line.split(' ')
            word = values[0].replace('-', '').lower()
            coefs = np.asarray(values[1:], dtype='float32')
            glove_db[word] = coefs
    
    ## Set Embeddings ##
    
    for word, i in tokenizer.word_index.items():
        
        embedding_vector = glove_db.get(word)
        
        if embedding_vector is not None:
            
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix, glove_db


# In[6]:


if __name__ == "__main__":
    
    headlines, tick_hists, effects = make_headline_to_effect_data()
    
    encoded_headlines, toke = encode_sentences(headlines, max_length=max_length, vocab_size=vocab_size)
    
    vocab_size = len(toke.word_counts)
    
    emb_matrix, glove_db = get_embedding_matrix(toke)

