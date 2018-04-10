
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
max_length  = 600
vocab_size  = None # Set by tokenizer
emb_size    = 300

model_type  = 'multireg'

epochs      = 100
batch_size  = 64

test_cutoff = datetime(2018, 2, 14)


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
    all_headlines, all_tick_hist, all_effects, test_indexes = [], [], [], []
    
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
                    
                effect = [(result_tick - previous_tick) / previous_tick / 0.023]

                for i in range(num_samples):

                    indexes = np.random.choice(np.arange(len(headlines)), sample_size, replace=False, p=probs)
                    
                    sample = [headlines[i] for i in indexes]
                    
                    if event_date > test_cutoff: # Mark as Test Example
                        test_indexes.append(len(all_headlines))

                    all_headlines.append(sample)
                    all_tick_hist.append(tick_hist)
                    all_effects.append(effect)
                    
    return all_headlines, np.array(all_tick_hist), np.array(all_effects), np.array(test_indexes)


# In[4]:


def encode_sentences(headlines, tokenizer=None, max_length=100):
    """
    Encoder
    
    Takes a list of headlines and converts them into vectors
    """
    ## Encoding Sentences
    
    sentences = []
    
    for example in headlines:
        sentences.append(" ".join([data[2] for data in example])) # Merge headlines into one long headline
        
    # print(np.mean(sizes))
    
    if not tokenizer:
        
        tokenizer = Tokenizer(filters='', lower=False) # Already PreProcessed
    
        tokenizer.fit_on_texts(sentences)
    
    encoded_headlines = tokenizer.texts_to_sequences(sentences)
    
    padded_headlines = pad_sequences(encoded_headlines, maxlen=max_length, padding='post')
    
    ## Encoding Meta Data
    
    # TODO
    
    return padded_headlines, tokenizer


# In[5]:


def split_data(X, X2, Y, test_indexes):
    """
    Splits X/Y to Train/Test
    """
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    
    train_indexes = np.setdiff1d(indexes, test_indexes, assume_unique=True)
    
    trainX,  testX  = X[train_indexes],  X[test_indexes]
    trainX2, testX2 = X2[train_indexes], X2[test_indexes]
    trainY,  testY  = Y[train_indexes],  Y[test_indexes]
    
    return trainX, trainX2, trainY, testX, testX2, testY


# In[6]:


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

def correct_sign_acc(y_true, y_pred):
    """
    Accuracy of Being Positive or Negative
    """
    diff = K.equal(y_true > 0, y_pred > 0)
    
    return K.mean(diff, axis=-1)

def get_model(emb_matrix):
    
    ## Headline ##
    
    headline_input = Input(shape=(max_length,), name="headlines")
    
    emb = Embedding(vocab_size + 1, emb_size, input_length=max_length, weights=[emb_matrix], trainable=True)(headline_input)
    emb = SpatialDropout1D(.2)(emb)
    
    # (TODO) MERGE META WITH EMBEDDINGS
    
    text_rnn = LSTM(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(emb)
    text_rnn = Activation('selu')(text_rnn)
    text_rnn = BatchNormalization()(text_rnn)
    
    text_rnn = LSTM(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=False)(text_rnn)
    text_rnn = Activation('selu')(text_rnn)
    text_rnn = BatchNormalization()(text_rnn)
    
    ## Ticks ##
    
    tick_input = Input(shape=(tick_window, 5), name="stockticks")
    
    tick_conv = Conv1D(filters=64, kernel_size=5, padding='same', activation='selu')(tick_input)
    tick_conv = MaxPooling1D(pool_size=2)(tick_conv)
    
    tick_rnn = LSTM(200, dropout=0.3, recurrent_dropout=0.3, return_sequences=False)(tick_conv)
    tick_rnn = Activation('selu')(tick_rnn)
    tick_rnn = BatchNormalization()(tick_rnn)
    
    ## Combined ##
    
    merged = concatenate([text_rnn, tick_rnn])
    
    final_dense = Dense(400)(merged)
    final_dense = Activation('selu')(final_dense)
    final_dense = BatchNormalization()(final_dense)
    final_dense = Dropout(0.5)(final_dense)
    
    final_dense = Dense(200)(merged)
    final_dense = Activation('selu')(final_dense)
    final_dense = BatchNormalization()(final_dense)
    final_dense = Dropout(0.5)(final_dense)
        
    pred_dense = Dense(1)(final_dense)
    out = pred_dense
        
    model = Model(inputs=[headline_input, tick_input], outputs=out)
    
    model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=[correct_sign_acc])
    
    return model


# In[7]:


if __name__ == "__main__":
    
    headlines, tick_hists, effects, test_indexes = make_headline_to_effect_data()
     
    encoded_headlines, toke = encode_sentences(headlines, max_length=max_length)
    
    vocab_size = len(toke.word_counts)
    
    emb_matrix, glove_db = get_embedding_matrix(toke)
    
    trainX, trainX2, trainY, testX, testX2, testY = split_data(encoded_headlines, tick_hists, effects, test_indexes)
    
    print(trainX.shape, trainX2.shape, testY.shape)


# In[ ]:

# TRAIN MODEL

if __name__ == "__main__":  
    
    ## Save Tokenizer ##
    
    with open(os.path.join('..', 'models', 'toke2-tick.pkl'), 'wb') as toke_file:
        pickle.dump(toke, toke_file, protocol=pickle.HIGHEST_PROTOCOL)
        
    ## Create Model ##
    
    model = get_model(emb_matrix)
    
    monitor_mode = 'correct_sign_acc'
    
    tensorboard = TensorBoard(log_dir="logs/{}".format(datetime.now().strftime("%Y,%m,%d-%H,%M,%S,tick," + model_type)))
    e_stopping = EarlyStopping(monitor='val_loss', patience=50)
    checkpoint = ModelCheckpoint(os.path.join('..', 'models', 'media-headlines-ticks-' + model_type + '.h5'), 
                                 monitor=monitor_mode,
                                 verbose=0,
                                 save_best_only=True)
    
    plot_model(model, to_file='model.png', show_shapes=True)
    
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
            

