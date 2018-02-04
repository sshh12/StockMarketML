
# coding: utf-8

# In[1]:

# Imports

from datetime import datetime, timedelta

import numpy as np
import os

import matplotlib.pyplot as plt

from keras.models import Sequential, load_model, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, concatenate, SpatialDropout1D
from keras.layers import Dense, Flatten, Embedding, LSTM, Activation, BatchNormalization, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# In[2]:

# Options

stocks = ['AAPL', 'AMD', 'AMZN', 'GOOG', 'MSFT']

max_length = 40
vocab_size = 5000
emb_size   = 256

epochs     = 120
batch_size = 32


# In[3]:


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


# In[4]:


def get_headline_data(stocks):
    """
    Headline Data
    
    This reads the headlines from the headline csv file (created by CollectData)
    """
    history = {}
    
    with open(os.path.join('..', 'data', "_".join(stocks) + '-headlines.csv'), 'r', encoding="utf8") as data:
        
        for line in data:

            if len(line) > 6:

                stock, date, headlines = line.split(",")
                
                headlines = eval(headlines.strip().replace('@', ','))
        
                if not stock in history:
                    
                    history[stock] = {}
                
                history[stock][date] = headlines
                
    return history


# In[5]:


def make_headline_to_effect_data(tick_data, head_data):
    """
    Headline -> Effect
    
    Creates essentially the X, Y data for the embedding model to use
    when analyzing/encoding headlines. Returns a list of headlines and
    a list of corresponding 'effects' which represent a change in the stock price.
    """
    sources, all_headlines, effects = [], [], []
    
    for stock, dates in head_data.items():
        
        for date, headlines in dates.items():
            
            ## Find Matching tick data dates for headline dates ##
            
            event_date = datetime.strptime(date, '%Y-%m-%d') # The date `of` headline
            effect_date = event_date + timedelta(days=1)     # The day after `affected` by headline
            
            for i in range(4):
                if event_date.strftime('%Y-%m-%d') in tick_data[stock]:
                    break
                else:
                    event_date -= timedelta(days=1)
            else:
                continue
                    
            for i in range(3):
                if effect_date.strftime('%Y-%m-%d') in tick_data[stock]:
                    break
                else:
                    effect_date += timedelta(days=1)
            else:
                continue
                
            event_date = event_date.strftime('%Y-%m-%d')
            effect_date = effect_date.strftime('%Y-%m-%d')
            
            ## Determine Effect ##
            
            if event_date in tick_data[stock] and effect_date in tick_data[stock]:
                
                tick_on = tick_data[stock][event_date]
                tick_after = tick_data[stock][effect_date]
                
                if tick_after[3] >= tick_on[3]: # Compare Close Prices
                    
                    effect = [1., 0.]
                    
                else:
                    
                    effect = [0., 1.]
                    
                for source, headline in headlines.items():

                    all_headlines.append(headline)
                    effects.append(effect)
                    sources.append(source)
                
    return sources, all_headlines, np.array(effects)


# In[6]:


def encode_sentences(sources, sentences, tokenizer=None, max_length=100, vocab_size=100):
    """
    Encoder
    
    Takes a list of headlines and converts them into vectors
    """
    ## Encoding Sentences
    
    if not tokenizer:
        
        tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False) # Preprocessed
    
        tokenizer.fit_on_texts(sentences)
    
    encoded_headlines = tokenizer.texts_to_sequences(sentences)
    
    padded_headlines = pad_sequences(encoded_headlines, maxlen=max_length, padding='post')
    
    ## Encoding Source
    
    source_set, source_mat = list(set(sources)), []
    
    for source in sources:
        
        row = [0] * len(source_set)
        row[source_set.index(source)] = 1
        source_mat.append(row)
        
    source_mat = np.array(source_mat)
    
    return source_mat, padded_headlines, tokenizer


# In[7]:


def split_data(X, X2, Y, ratio):
    """
    Splits X/Y to Train/Test
    """
    train_size = int(len(X) * ratio)
    
    trainX,  testX  = X[:train_size],  X[train_size:]
    trainX2, testX2 = X2[:train_size], X2[train_size:]
    trainY,  testY  = Y[:train_size],  Y[train_size:]
        
    indexes = np.arange(trainX.shape[0])
    np.random.shuffle(indexes)
        
    trainX  = trainX[indexes]
    trainX2 = trainX2[indexes]
    trainY  = trainY[indexes]
    
    return trainX, trainX2, trainY, testX, testX2, testY


# In[8]:


def get_model():
    
    ## Text
    
    text_input = Input(shape=(max_length,))
    
    emb = Embedding(vocab_size, emb_size, input_length=max_length)(text_input)
    emb = SpatialDropout1D(.3)(emb)
    
    # conv = Conv1D(filters=64, kernel_size=5, padding='same', activation='selu')(emb)
    # conv = MaxPooling1D(pool_size=3)(conv)
    
    lstm = LSTM(300, dropout=0.3, recurrent_dropout=0.3)(emb)
    # lstm = Activation('selu')(lstm)
    # lstm = BatchNormalization()(lstm)
    
    ## Source
    
    source_input = Input(shape=(4,))
    
    ## Combined
    
    merged = concatenate([lstm, source_input])
    
    dense_1 = Dense(200)(merged)
    dense_1 = Activation('relu')(dense_1)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    
    dense_2 = Dense(100)(dense_1)
    dense_2 = Activation('relu')(dense_2)
    dense_2 = BatchNormalization()(dense_2)
    dense_2 = Dropout(0.5)(dense_2)
    
    dense_3 = Dense(100)(dense_2)
    dense_3 = Activation('relu')(dense_3)
    dense_3 = BatchNormalization()(dense_3)
    dense_3 = Dropout(0.5)(dense_3)
    
    dense_4 = Dense(2)(dense_3)
    out = Activation('softmax')(dense_4)
    
    model = Model(inputs=[text_input, source_input], outputs=out)
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    
    return model


# In[9]:


if __name__ == "__main__":
    
    tick_data = get_tick_data(stocks)
    head_data = get_headline_data(stocks)
    
    sources, headlines, effects = make_headline_to_effect_data(tick_data, head_data)
    
    encoded_sources, encoded_headlines, toke = encode_sentences(sources, 
                                                                headlines, 
                                                                max_length=max_length, 
                                                                vocab_size=vocab_size)
    
    trainX, trainX2, trainY, testX, testX2, testY = split_data(encoded_headlines, encoded_sources, effects, .8)
    
    print(trainX.shape, trainX2.shape, testY.shape)


# In[10]:


if __name__ == "__main__":
    
    model = get_model()
    
    e_stopping = EarlyStopping(monitor='val_loss', patience=70)
    checkpoint = ModelCheckpoint(os.path.join('..', 'models', 'media-headlines.h5'), 
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    
    history = model.fit([trainX, trainX2],
                        trainY,
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=([testX, testX2], testY),
                        verbose=0,
                        callbacks=[e_stopping, checkpoint])
    
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.legend(['LogTrainLoss', 'LogTestLoss'])
    plt.show()
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['TrainAcc', 'TestAcc'])
    plt.show()
    


# In[11]:


if __name__ == "__main__":
    
    model = load_model(os.path.join('..', 'models', 'media-headlines.h5'))
    
    test_sents = [
        'the ceo of **COMPANY** was fired after selling a bad **PRODUCT**', 
        '**COMPANY** just released a **PRODUCT** thats better than every other company',
        '**COMPANY**s **PRODUCT** killed a family of ducks in a sensor malfunction',
        'the **COMPANY** team released a breakthrough in **PRODUCT** gaming'
    ]
    
    encoded_sources, test_encoded, _ = encode_sentences(['reuters', 'twitter', 'reddit', 'seekingalpha'], 
                                                        test_sents, 
                                                        tokenizer=toke, 
                                                        max_length=max_length, 
                                                        vocab_size=vocab_size)
    
    predictions = model.predict([test_encoded, encoded_sources])
    
    for i in range(len(test_sents)):
        
        print("")
        print(test_sents[i])
        print(predictions[i])
        print("Stock Will Go Up" if np.argmax(predictions[i]) == 0 else "Stock Will Go Down")

