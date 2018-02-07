
# coding: utf-8

# In[1]:

# Imports

from datetime import datetime, timedelta

from Database import db

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
sources = ['reddit', 'twitter', 'seekingalpha']

max_length  = 40
vocab_size  = 5000
emb_size    = 256

epochs     = 120
batch_size = 32


# In[3]:


def make_headline_to_effect_data():
    """
    Headline -> Effect
    
    Creates essentially the X, Y data for the embedding model to use
    when analyzing/encoding headlines. Returns a list of headlines and
    a list of corresponding 'effects' which represent a change in the stock price.
    """
    sources, headlines, effects = [], [], []
    
    with db() as (conn, cur):
        
        for stock in stocks:
            
            print("Fetching..." + stock, end="\r")
            
            ## Go through all the headlines ##
            
            cur.execute("SELECT date, source, content FROM headlines WHERE stock=? AND LENGTH(content) >= 16", [stock])
            headline_query = cur.fetchall()
            
            for (date, source, content) in headline_query:
                
                event_date = datetime.strptime(date, '%Y-%m-%d') # The date of headline
                
                ## Find corresponding tick data ## 
                
                cur.execute("""SELECT adjclose FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date""", 
                            [stock, (event_date - timedelta(days=3)).strftime('%Y-%m-%d'), event_date.strftime('%Y-%m-%d')])
                
                before_headline_ticks = cur.fetchall()
                
                cur.execute("""SELECT adjclose FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date""", 
                            [stock, (event_date + timedelta(days=1)).strftime('%Y-%m-%d'), (event_date + timedelta(days=4)).strftime('%Y-%m-%d')])
                
                after_headline_ticks = cur.fetchall()
                
                ## Create training example ##
                
                if len(before_headline_ticks) > 0 and len(after_headline_ticks) > 0:
                    
                    previous_tick = before_headline_ticks[-1]
                    result_tick = after_headline_ticks[0]
                
                    if result_tick > previous_tick:
                        
                        effect = [1., 0.]
                        
                    else:
                        
                        effect = [0., 1.]
                        
                    sources.append(source)
                    headlines.append(content)
                    effects.append(effect)
                    
    return sources, headlines, np.array(effects)


# In[4]:


def encode_sentences(sources, sentences, tokenizer=None, max_length=100, vocab_size=100):
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
    
    ## Encoding Source
    
    source_mat = []
    
    for source in sources:
        
        row = [0] * len(sources)
        row[sources.index(source)] = 1
        source_mat.append(row)
        
    source_mat = np.array(source_mat)
    
    return source_mat, padded_headlines, tokenizer


# In[5]:


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


# In[6]:


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
    
    source_input = Input(shape=(len(sources),))
    
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
    
    dense_3 = Dense(2)(dense_2)
    out = Activation('softmax')(dense_3)
    
    model = Model(inputs=[text_input, source_input], outputs=out)
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    
    return model


# In[7]:


if __name__ == "__main__":
    
    sources, headlines, effects = make_headline_to_effect_data()
    
    encoded_sources, encoded_headlines, toke = encode_sentences(sources, 
                                                                headlines, 
                                                                max_length=max_length, 
                                                                vocab_size=vocab_size)
    
    trainX, trainX2, trainY, testX, testX2, testY = split_data(encoded_headlines, encoded_sources, effects, .8)
    
    print(trainX.shape, trainX2.shape, testY.shape)


# In[8]:


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
    


# In[10]:


if __name__ == "__main__":
    
    model = load_model(os.path.join('..', 'models', 'media-headlines.h5'))
    
    test_sents = [
        'the ceo of **COMPANY** was fired after selling a bad **PRODUCT**', 
        '**COMPANY** just released a **PRODUCT** thats better than every other company',
        '**COMPANY**s **PRODUCT** killed a family of ducks in a sensor malfunction',
        'the **COMPANY** team released a breakthrough in **PRODUCT** gaming'
    ]
    
    encoded_sources, test_encoded, _ = encode_sentences(['twitter', 'twitter', 'reddit', 'seekingalpha'], 
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

