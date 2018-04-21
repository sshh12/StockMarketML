
# coding: utf-8

# In[48]:

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

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# In[54]:

# Options

stocks      = ['AMD', 'INTC']
all_sources = ['reddit', 'reuters', 'twitter', 'seekingalpha', 'fool', 'wsj', 'thestreet']

model_type  = 'multiheadline'

doc2vec_options = dict(
    size=300, 
    window=10, 
    min_count=5, 
    workers=10,
    alpha=0.025, 
    min_alpha=0.025, 
    max_vocab_size=15000
)

tick_window = 20

test_cutoff = datetime(2018, 3, 20)


# In[63]:


def add_time(date, days):
    
    return (date + timedelta(days=days)).strftime('%Y-%m-%d')

def clean(sentence):
    
    sentence = sentence.lower()
    sentence = sentence.replace('-', ' ').replace('_', ' ').replace('&', ' ')
    sentence = ''.join(c for c in sentence if c in "abcdefghijklmnopqrstuvwxyz ")
    sentence = re.sub('\s+', ' ', sentence)
    
    return sentence.strip()

def make_doc_embeddings():

    docs, labels = [], []
    
    class LabeledLineSentence:
        
        def __init__(self, docs, labels):
            self.docs = docs
            self.labels = labels
            
        def __iter__(self):
            for idx, doc in enumerate(self.docs):
                yield TaggedDocument(doc.split(), [self.labels[idx]])
    
    with db() as (conn, cur):
        
        for stock in stocks:
            
            ## Headline For Every Date ##
            
            cur.execute("SELECT DISTINCT date FROM headlines WHERE stock=? ORDER BY date ASC", [stock])
            dates = [date[0] for date in cur.fetchall()]
            
            for date in tqdm_notebook(dates, desc=stock):
                
                ## Collect Headlines ##
                
                event_date = datetime.strptime(date, '%Y-%m-%d')
                
                cur.execute("SELECT date, source, rawcontent FROM headlines WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date DESC", 
                            [stock, add_time(event_date, -14), date])
                headlines = [(date, source, clean(content), (event_date - datetime.strptime(date, '%Y-%m-%d')).days) 
                                 for (date, source, content) in cur.fetchall() if content]
                
                if len(headlines) < 2:
                    continue
                
                ## Create training example ##
                    
                contents = [headline[2] for headline in headlines]

                doc = " ".join(contents)
                
                docs.append(doc)
                labels.append(stock + " " + date)
            
    doc_iter = LabeledLineSentence(docs, labels)
            
    vec_model = Doc2Vec(documents=doc_iter, **doc2vec_options)
    
    vectors = {stock: {} for stock in stocks}
    
    for label in labels:
        
        stock, date = label.split(" ")
        
        vectors[stock][date] = vec_model.docvecs[label]
                    
    return vec_model, vectors, (docs, labels)

def make_tick_data():
    
    tick_vecs = {stock: {} for stock in stocks}
    effect_vecs = {stock: {} for stock in stocks}
    
    with db() as (conn, cur):
        
        for stock in stocks:
            
            cur.execute("SELECT DISTINCT date FROM headlines WHERE stock=? ORDER BY date ASC LIMIT 1", [stock])
            start_date = cur.fetchall()[0][0]
            
            cur.execute("SELECT DISTINCT date FROM ticks WHERE stock=? AND date >= ? ORDER BY date ASC", [stock, start_date])
            dates = [date[0] for date in cur.fetchall()]
            
            for date in dates:
                
                event_date = datetime.strptime(date, '%Y-%m-%d') # The date of headline

                ## Find corresponding tick data ## 

                cur.execute("""SELECT open, high, low, adjclose, volume FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date DESC LIMIT 52""", 
                            [stock, 
                             add_time(event_date, -80), 
                             add_time(event_date, 0)])

                before_headline_ticks = cur.fetchall()

                if len(before_headline_ticks) < tick_window:
                    continue

                cur.execute("""SELECT adjclose FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date ASC LIMIT 1""", 
                            [stock, 
                            add_time(event_date, 1), 
                            add_time(event_date, 4)])

                after_headline_ticks = cur.fetchall()

                ## Create ##

                if len(after_headline_ticks) == 0:
                    continue

                window_ticks = np.array(list(reversed(before_headline_ticks[:tick_window]))) # Flip so in chron. order
                fifty_ticks = np.array(before_headline_ticks) # Use last 50 ticks to normalize

                previous_tick = before_headline_ticks[0][3]
                result_tick = after_headline_ticks[0][0]

                if previous_tick and result_tick:

                    window_ticks -= np.mean(fifty_ticks, axis=0)
                    window_ticks /= np.std(fifty_ticks, axis=0)

                    # Percent Diff (/ Normalization Constant)
                    effect = [(result_tick - previous_tick) / previous_tick / 0.023]
                    
                    tick_vecs[stock][date] = window_ticks
                    effect_vecs[stock][date] = effect
                    
    return tick_vecs, effect_vecs


# In[68]:


def merge_data(doc_vecs, tick_vecs, effect_vecs):
    
    X, Y = [], []
    
    for stock in stocks:
        
        for date, tick_vec in tick_vecs[stock].items():
            
            x = []
            y = effect_vecs[stock][date]
            
            event_date = datetime.strptime(date, '%Y-%m-%d')
            
            window_dates = [add_time(event_date, -i) for i in range(tick_window)]
            
            for i in range(tick_window):
                
                if window_dates[i] not in doc_vecs[stock]:
                    break
                    
                x_i = np.concatenate([tick_vec[i], doc_vecs[stock][window_dates[i]]])
                
                x.append(x_i)
                
            if len(x) == tick_window:
                X.append(x)
                Y.append(y)
        
    return X, Y


# In[ ]:


def correct_sign_acc(y_true, y_pred):
    """
    Accuracy of Being Positive or Negative
    """
    diff = K.equal(y_true > 0, y_pred > 0)
    
    return K.mean(diff, axis=-1)

def get_model():
    
    pass


# In[64]:

# Load Data

if __name__ == "__main__":
    
    vec_model, doc_vecs, doc_data = make_doc_embeddings() #vec_model.docvecs.most_similar("INTC 2016-04-20")
    
    tick_vecs, effect_vecs = make_tick_data()
    
    X, Y = merge_data(doc_vecs, tick_vecs, effect_vecs)


# In[ ]:

# TRAIN MODEL

if __name__ == "__main__":  

    ## Create Model ##
    
    model = get_model()
    
    monitor_mode = 'correct_sign_acc'
    
    #tensorboard = TensorBoard(log_dir="logs/{}".format(datetime.now().strftime("%Y,%m,%d-%H,%M,%S,tick," + model_type)))
    e_stopping = EarlyStopping(monitor='val_loss', patience=50)
    #checkpoint = ModelCheckpoint(os.path.join('..', 'models', 'media-headlines-ticks-' + model_type + '.h5'), 
    #                             monitor=monitor_mode,
    #                             verbose=0,
    #                             save_best_only=True)
    
    #plot_model(model, to_file='model.png', show_shapes=True)
    
    ## Train ##
    
    history = model.fit([trainX, trainX2],
                        trainY,
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=([testX, testX2], testY),
                        verbose=0,
                        callbacks=[e_stopping, checkpoint, tensorboard])
    
    ## Display Train History ##
    
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.legend(['LogTrainLoss', 'LogTestLoss'])
    plt.show()
    
    plt.plot(history.history[monitor_mode])
    plt.plot(history.history['val_' + monitor_mode])
    plt.legend(['TrainAcc', 'TestAcc'])
    plt.show()


# In[9]:

# Predict (TEST)

def predict(stock, model=None, toke=None, current_date=None, predict_date=None):
    
    import keras.metrics
    keras.metrics.correct_sign_acc = correct_sign_acc
    
    if not model or not toke:
        
        with open(os.path.join('..', 'models', 'toke2-tick.pkl'), 'rb') as toke_file:
            toke = pickle.load(toke_file)
    
        model = load_model(os.path.join('..', 'models', 'media-headlines-ticks-' + model_type + '.h5'))
        
    vocab_size = len(toke.word_counts)
        
    if not current_date:
        current_date = datetime.today()
        
    if not predict_date:
        predict_date = current_date + timedelta(days=1)
    
    all_headlines, all_tick_hist = [], []
    
    with db() as (conn, cur):
        
        event_date = current_date
        date = datetime.strftime(event_date, '%Y-%m-%d')
                
        cur.execute("SELECT date, source, rawcontent FROM headlines WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date DESC", 
                    [stock, add_time(event_date, -14), date])
        headlines = [(date, source, clean(content), (event_date - datetime.strptime(date, '%Y-%m-%d')).days) 
                        for (date, source, content) in cur.fetchall() if content]
                    
        ## Find corresponding tick data ## 
                
        cur.execute("""SELECT open, high, low, adjclose, volume FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date DESC""", 
                    [stock, 
                     add_time(event_date, -30 - tick_window), 
                     add_time(event_date, 0)])
                
        before_headline_ticks = cur.fetchall()[:tick_window]
        actual_current = before_headline_ticks[0][3]
                
        tick_hist = np.array(before_headline_ticks)
        tick_hist -= np.mean(tick_hist, axis=0)
        tick_hist /= np.std(tick_hist, axis=0)
                
        ## Create training example ##

        probs = [1 / (headline[3] + 1) for headline in headlines]
        probs /= np.sum(probs)
                    
        contents = [headline[2] for headline in headlines]

        num_samples = len(contents) // sample_size

        for i in range(num_samples):

            indexes = np.random.choice(np.arange(len(headlines)), sample_size, replace=False, p=probs)
                    
            sample = [headlines[i] for i in indexes]

            all_headlines.append(sample)
            all_tick_hist.append(tick_hist)
        
        ## Process ##
    
        encoded_headlines, toke = encode_sentences(all_headlines, tokenizer=toke, max_length=max_length)
        
        tick_hists = np.array(all_tick_hist)
        
        predictions = model.predict([encoded_headlines, tick_hists])[:, 0]
        
        prices = predictions * 0.023 * actual_current + actual_current
        
        return predictions, prices
    


# In[11]:

# [TEST] Spot Testing

if __name__ == "__main__":
    
    ## **This Test May Overlap w/Train Data** ##
    
    ## Options ##
    
    stock = 'INTC'
    current_date = '2018-03-07'
    predict_date = '2018-03-08'
    
    ## Run ##
    
    predictions, prices = predict(stock, 
                                  current_date=datetime.strptime(current_date, '%Y-%m-%d'), 
                                  predict_date=datetime.strptime(predict_date, '%Y-%m-%d'))
    
    ## Find Actual Value ##
     
    with db() as (conn, cur):
    
        cur.execute("""SELECT adjclose FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date ASC LIMIT 1""", 
                        [stock, 
                        add_time(datetime.strptime(predict_date, '%Y-%m-%d'), 0), 
                        add_time(datetime.strptime(predict_date, '%Y-%m-%d'), 6)])

        after_headline_ticks = cur.fetchall()
        try:
            actual_result = after_headline_ticks[0][0]
        except:
            actual_result = -1
            
    ## Display ##
            
    parse = lambda num: str(round(num, 2))
    
    print("Predicting Change Coef: " + parse(np.mean(predictions)))
    print("Predicting Price: " + parse(np.mean(prices)))
    print("Actual Price: " + parse(actual_result))
            

