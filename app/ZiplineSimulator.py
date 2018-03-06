
# coding: utf-8

# In[2]:

# Imports

from contextlib import contextmanager
from datetime import datetime, timedelta
import sqlite3
import os

from zipline.api import order, order_target, record, symbol
from zipline.finance import commission, slippage
import zipline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:

# Access Database

@contextmanager
def db(db_filename='stock.db'):
    
    conn = sqlite3.connect(os.path.join('..', 'data', db_filename), detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

    cur = conn.cursor()
    
    yield conn, cur
    
    conn.close()
    


# In[ ]:

# Custom Slippage Model

class TradeNearTheOpenSlippageModel(slippage.SlippageModel):

    def __init__(self, deviation=0.001):
        
        self.deviation = deviation

    def process_order(self, data, order):
        
        rand = min(np.abs(np.random.normal(0, self.deviation)), 1) # Generate a random value thats likely zero (zero=openprice)
        
        open_price = data.current(symbol(stock), 'open') 
        close_price = data.current(symbol(stock), 'close') 
        
        new_price = (close_price - open_price) * rand + open_price 
 
        return (new_price, order.amount)  


# In[ ]:

# !+!+!+!+! PLACE MODEL HERE !+!+!+!+!

from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.models import load_model

stocks      = ['AAPL', 'AMD', 'AMZN', 'GOOG', 'MSFT', 'INTC']
all_sources = ['reddit', 'reuters', 'twitter', 'seekingalpha', 'fool', 'wsj', 'thestreet']

tick_window = 30
max_length  = 50
vocab_size  = None # Set by tokenizer
emb_size    = 300

model_type  = 'regression'

epochs      = 250
batch_size  = 128


def correct_sign_acc(y_true, y_pred):
    """
    Accuracy of Being Positive or Negative
    """
    diff = K.equal(y_true > 0, y_pred > 0)
    
    return K.mean(diff, axis=-1)

import keras.metrics
keras.metrics.correct_sign_acc = correct_sign_acc

import pickle

with open(os.path.join('..', 'models', 'toke-tick.pkl'), 'rb') as toke_file:
    toke = pickle.load(toke_file)
    
model = load_model(os.path.join('..', 'models', 'media-headlines-ticks-' + model_type + '.h5'))


# In[ ]:

# !+!+!+!+! PLACE MODEL HERE !+!+!+!+!

def add_time(date, days):
    
    return (date + timedelta(days=days)).strftime('%Y-%m-%d')

def encode_sentences(meta, sentences, tokenizer=None, max_length=100, vocab_size=100):
    """
    Encoder
    
    Takes a list of headlines and converts them into vectors
    """
    ## Encoding Sentences
    
    if not tokenizer:
        
        tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False) # Already Preprocessed
    
        tokenizer.fit_on_texts(sentences)
    
    encoded_headlines = tokenizer.texts_to_sequences(sentences)
    
    padded_headlines = pad_sequences(encoded_headlines, maxlen=max_length, padding='post')
    
    ## Encoding Meta Data
    
    # OneHot(Source [reddit/twitter/reuters etc..]) + OneHot(WeekDay)
    
    meta_matrix = np.zeros((len(sentences), len(all_sources) + 7))
    index = 0
    
    for (source, weekday) in meta:
        
        meta_matrix[index, all_sources.index(source)] = 1
        meta_matrix[index, len(all_sources) + weekday] = 1
        
        index += 1
    
    return meta_matrix, padded_headlines, tokenizer


# In[ ]:

# !+!+!+!+! PLACE MODEL HERE !+!+!+!+!

def predict(stock, model=None, toke=None, current_date=None, predict_date=None, look_back=None):
    
    if not model or not toke:
        
        with open(os.path.join('..', 'models', 'toke-tick.pkl'), 'rb') as toke_file:
            toke = pickle.load(toke_file)
    
        model = load_model(os.path.join('..', 'models', 'media-headlines-ticks-' + model_type + '.h5'))
        
    vocab_size = len(toke.word_counts)
        
    if not current_date:
        current_date = datetime.today()
        
    if not predict_date:
        predict_date = current_date + timedelta(days=1)
        
    if not look_back:
        look_back = 3
    
    pretick_date = add_time(current_date, -look_back)
    
    with db() as (conn, cur):
        
        ## Select Actual Stock Values ##
                
        cur.execute("""SELECT open, high, low, adjclose, volume FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date DESC""", 
                    [stock, 
                    add_time(current_date, -30 - tick_window), 
                    add_time(current_date, 0)])
                
        before_headline_ticks = cur.fetchall()[:tick_window]
        actual_current = before_headline_ticks[0][3]
        
        cur.execute("""SELECT adjclose FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date ASC LIMIT 1""", 
                    [stock, 
                    add_time(predict_date, 1), 
                    add_time(predict_date, 5)])
        
        after_headline_ticks = cur.fetchall()
        
        tick_hist = np.array(before_headline_ticks)
        tick_hist -= np.mean(tick_hist, axis=0)
        tick_hist /= np.std(tick_hist, axis=0)
        
        ## Find Headlines ##
    
        cur.execute("SELECT date, source, content FROM headlines WHERE date BETWEEN ? AND ? AND stock=?", [pretick_date, current_date, stock])
        headlines = cur.fetchall()
        
        ## Process ##
        
        meta, test_sents = [], []
        
        for (date, source, content) in headlines:
            
            meta.append([source, datetime.strptime(date, '%Y-%m-%d').weekday()])
            test_sents.append(content)
            
        encoded_meta, test_encoded, _ = encode_sentences(meta, 
                                                         test_sents, 
                                                         tokenizer=toke, 
                                                         max_length=max_length,
                                                         vocab_size=vocab_size)
        
        tick_hists = np.array([tick_hist] * len(headlines))
        
        predictions = model.predict([test_encoded, tick_hists, encoded_meta])[:, 0]
        
        prices = predictions * 0.023 * actual_current + actual_current
        
        return predictions, prices


# In[ ]:

# Predictors

def predict_perfect(stock, date): # ~Perfect~ Predictor
    
    with db() as (conn, cur):
        
        cur.execute("SELECT date, adjclose FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date DESC LIMIT 1", 
                    [stock, (date + timedelta(days=-5)).strftime('%Y-%m-%d'), (date + timedelta(days=0)).strftime('%Y-%m-%d')])
        before = cur.fetchall()[0]
        
        cur.execute("SELECT date, adjclose FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date ASC LIMIT 1", 
                    [stock, (date + timedelta(days=1)).strftime('%Y-%m-%d'), (date + timedelta(days=5)).strftime('%Y-%m-%d')])
        after = cur.fetchall()[0]
        
        if after[1] > before[1]:
            return 999
        else:
            return -999
        
def predict_deep_nn(stock, date):
    
    preds, prices = predict(stock, model, toke, current_date=date)

# predict_deep_nn('AAPL', datetime(2017, 10, 15))


# In[ ]:

stock = 'AMD'

def initialize(context):
    context.set_commission(commission.PerShare(cost=0, min_trade_cost=1.0))
    context.set_slippage(TradeNearTheOpenSlippageModel())
    
def handle_data(context, data):
    
    date = data.current(symbol(stock), 'last_traded').to_datetime()
    
    pred = predict_perfect(stock, date + timedelta(days=0))
    
    shares = context.portfolio.positions[symbol(stock)].amount
    
    # print(date, data.current(symbol(stock), 'price'), pred, shares, round(context.portfolio.cash))
    
    if pred > 0:
        max_shares = context.portfolio.cash // data.current(symbol(stock), 'price')
        if max_shares > 0:
            order(symbol(stock), max_shares)
    else:
        if shares > 0:
            order_target(symbol(stock), 0)
        
    record(stock=data.current(symbol(stock), 'price'))
    record(shares=context.portfolio.positions[symbol(stock)].amount)

start = pd.to_datetime('2017-01-01').tz_localize('US/Eastern')
end = pd.to_datetime('2018-03-01').tz_localize('US/Eastern')

perf = zipline.run_algorithm(start, end, initialize, 100, handle_data=handle_data)


# In[ ]:

ax1 = plt.subplot(211)
perf.portfolio_value.plot(ax=ax1)
ax1.set_ylabel('Portfolio Value')
ax2 = plt.subplot(212, sharex=ax1)
perf.stock.plot(ax=ax2)
ax2.set_ylabel('Stock Price')
plt.show()


# In[ ]:

print(perf.columns)
perf[['period_open', 'period_close', 'starting_cash', 'ending_cash', 'portfolio_value', 'shares']].head()

