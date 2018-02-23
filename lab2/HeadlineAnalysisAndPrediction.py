
# coding: utf-8

# In[1]:

# Imports

from datetime import datetime, timedelta

from Database import db

import numpy as np
import pickle
import os

import matplotlib.pyplot as plt

from keras.optimizers import RMSprop
from keras.models import Sequential, load_model, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, concatenate, SpatialDropout1D, GRU
from keras.layers import Dense, Flatten, Embedding, LSTM, Activation, BatchNormalization, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
import keras.backend as K


# In[2]:

# Options

stocks      = ['AAPL', 'AMD', 'AMZN', 'GOOG', 'MSFT']
all_sources = ['reddit', 'reuters', 'twitter', 'seekingalpha', 'fool']

max_length  = 50
vocab_size  = None # Set by tokenizer
emb_size    = 300

model_type  = 'regression'

epochs      = 180
batch_size  = 32


# In[10]:


def make_headline_to_effect_data():
    """
    Headline -> Effect
    
    Creates essentially the X, Y data for the embedding model to use
    when analyzing/encoding headlines. Returns a list of headlines and
    a list of corresponding 'effects' which represent a change in the stock price.
    """
    meta, headlines, effects = [], [], []
    
    with db() as (conn, cur):
        
        for stock in stocks:
            
            print("Fetching Stock..." + stock)
            
            ## Go through all the headlines ##
            
            cur.execute("SELECT date, source, content FROM headlines WHERE stock=? AND LENGTH(content) >= 16", [stock])
            headline_query = cur.fetchall()
            
            for (date, source, content) in headline_query:
                
                event_date = datetime.strptime(date, '%Y-%m-%d') # The date of headline
                
                add_time = lambda e, days: (e + timedelta(days=days)).strftime('%Y-%m-%d')
                
                ## Find corresponding tick data ## 
                
                cur.execute("""SELECT AVG(adjclose) FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date""", 
                            [stock, 
                             add_time(event_date, -3), 
                             add_time(event_date, 0)])
                
                before_headline_ticks = cur.fetchall()
                
                cur.execute("""SELECT AVG(adjclose) FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date""", 
                            [stock, 
                             add_time(event_date, 1), 
                             add_time(event_date, 6)])
                
                after_headline_ticks = cur.fetchall()
                
                ## Create training example ##
                
                if len(before_headline_ticks) > 0 and len(after_headline_ticks) > 0 and before_headline_ticks[0][0] != None and after_headline_ticks[0][0] != None:
                    
                    previous_tick = before_headline_ticks[-1][0]
                    result_tick = after_headline_ticks[0][0]
                    
                    if model_type == 'regression':
                        
                        # Percent Diff (+Normalization Constant)
                        effect = [(result_tick - previous_tick) / previous_tick / 0.0044]
                    
                    else:
                
                        if result_tick > previous_tick:

                            effect = [1., 0.]

                        else:

                            effect = [0., 1.]
                        
                    meta.append((source, event_date.weekday()))
                    headlines.append(content)
                    effects.append(effect)
                    
    return meta, headlines, np.array(effects)


# In[4]:


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


# In[5]:


def split_data(X, X2, Y, ratio):
    """
    Splits X/Y to Train/Test
    """
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    
    X  = X[indexes]
    X2 = X2[indexes]
    Y  = Y[indexes]
    
    train_size = int(len(X) * ratio)
    
    trainX,  testX  = X[:train_size],  X[train_size:]
    trainX2, testX2 = X2[:train_size], X2[train_size:]
    trainY,  testY  = Y[:train_size],  Y[train_size:]
    
    return trainX, trainX2, trainY, testX, testX2, testY


# In[6]:


def get_embedding_matrix(tokenizer, pretrained_file='glove.840B.300d.txt', purge=False):
    """Load Vectors from Glove File"""
    print("Loading WordVecs...")
    
    ## Load Glove File (Super Slow) ##
    
    glove_db = dict()
    
    with open(os.path.join('..', 'data', pretrained_file), 'r', encoding="utf-8") as glove:

        for line in glove:

            values = line.split(' ')
            word = values[0].replace('-', '').lower()
            coefs = np.asarray(values[1:], dtype='float32')
            glove_db[word] = coefs

    print('Loaded WordVectors...' + str(len(glove_db)))
    
    ## Set Embeddings ##
    
    embedding_matrix = np.zeros((vocab_size + 1, emb_size))
    
    for word, i in tokenizer.word_index.items():
        
        embedding_vector = glove_db.get(word)
        
        if embedding_vector is not None:
            
            embedding_matrix[i] = embedding_vector
            
        elif purge:
            
            with db() as (conn, cur):
                
                cur.execute("SELECT 1 FROM dictionary WHERE word=?", [word])
                
                if len(cur.fetchall()) == 0:
                    
                    print("Purge..." + word)

                    cur.execute("DELETE FROM headlines WHERE content LIKE ?", ["%" + word + "%"])
                    conn.commit()
            
    return embedding_matrix, glove_db

def correct_sign_acc(y_true, y_pred):
    """
    Accuracy of Being Positive or Negative
    """
    diff = K.equal(y_true > 0, y_pred > 0)
    
    return K.mean(diff, axis=-1)

def get_model(emb_matrix):
    
    ## Headline ##
    
    headline_input = Input(shape=(max_length,))
    
    emb = Embedding(vocab_size + 1, emb_size, input_length=max_length, weights=[emb_matrix], trainable=True)(headline_input)
    emb = SpatialDropout1D(.2)(emb)
    
    conv = Conv1D(filters=64, kernel_size=5, padding='same', activation='selu')(emb)
    conv = MaxPooling1D(pool_size=3)(conv)
    
    text_rnn = LSTM(200, dropout=0.3, recurrent_dropout=0.3, return_sequences=False)(conv)
    text_rnn = Activation('selu')(text_rnn)
    text_rnn = BatchNormalization()(text_rnn)
    
    # text_rnn = LSTM(300, dropout=0.3, recurrent_dropout=0.3)(text_rnn)
    # text_rnn = Activation('relu')(text_rnn)
    # text_rnn = BatchNormalization()(text_rnn)
    
    ## Source ##
    
    meta_input = Input(shape=(len(all_sources) + 7,))
    
    ## Combined ##
    
    merged = concatenate([text_rnn, meta_input])
    
    final_dense = Dense(100)(merged)
    final_dense = Activation('selu')(final_dense)
    final_dense = BatchNormalization()(final_dense)
    final_dense = Dropout(0.5)(final_dense)
    
    final_dense = Dense(100)(merged)
    final_dense = Activation('selu')(final_dense)
    final_dense = BatchNormalization()(final_dense)
    final_dense = Dropout(0.5)(final_dense)
    
    if model_type == 'regression':
        
        pred_dense = Dense(1)(final_dense)
        out = pred_dense
        
        model = Model(inputs=[headline_input, meta_input], outputs=out)
    
        model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=[correct_sign_acc])
    
    else:
    
        pred_dense = Dense(2)(final_dense)
        out = Activation('softmax')(pred_dense)
        
        model = Model(inputs=[headline_input, meta_input], outputs=out)
    
        model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
    
    return model


# In[11]:


if __name__ == "__main__":
    
    meta, headlines, effects = make_headline_to_effect_data()
    
    encoded_meta, encoded_headlines, toke = encode_sentences(meta, 
                                                             headlines, 
                                                             max_length=max_length, 
                                                             vocab_size=vocab_size)
    
    vocab_size = len(toke.word_counts)
    print("Found Words......" + str(vocab_size))
    
    emb_matrix, glove_db = get_embedding_matrix(toke)
    
    trainX, trainX2, trainY, testX, testX2, testY = split_data(encoded_headlines, encoded_meta, effects, .8)
    
    print(trainX.shape, trainX2.shape, testY.shape)


# In[12]:

# TRAIN MODEL

if __name__ == "__main__": 
    
    ## Save Tokenizer ##
    
    with open(os.path.join('..', 'models', 'toke.pkl'), 'wb') as toke_file:
        pickle.dump(toke, toke_file, protocol=pickle.HIGHEST_PROTOCOL)
        
    ## Create Model ##
    
    model = get_model(emb_matrix)
    
    if model_type == 'regression':
        monitor_mode = 'correct_sign_acc'
    else:
        monitor_mode = 'val_acc'
    
    tensorboard = TensorBoard(log_dir="logs/{}".format(datetime.now().strftime("%Y,%m,%d-%H,%M,%S," + model_type)))
    e_stopping = EarlyStopping(monitor=monitor_mode, patience=60)
    checkpoint = ModelCheckpoint(os.path.join('..', 'models', 'media-headlines-' + model_type + '.h5'), 
                                 monitor=monitor_mode,
                                 verbose=0,
                                 save_best_only=True)
    
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
    


# In[13]:

# TEST MODEL

if __name__ == "__main__":
    
    ## Load Model For Manual Testing ##
    
    import keras.metrics
    keras.metrics.correct_sign_acc = correct_sign_acc
    
    with open(os.path.join('..', 'models', 'toke.pkl'), 'rb') as toke_file:
        toke = pickle.load(toke_file)
    
    model = load_model(os.path.join('..', 'models', 'media-headlines-' + model_type + '.h5'))
    
    ## **This Test May Overlap w/Train Data** ##
    
    pretick_date = '2018-02-12'
    current_date = '2018-02-14'
    predict_date = '2018-02-15'
    stock = 'AAPL'
    
    with db() as (conn, cur):
        
        ## Select Actual Stock Values ##
        
        cur.execute("""SELECT adjclose FROM ticks WHERE stock=? AND date BETWEEN ? AND ? ORDER BY date""", 
                    [stock, current_date, predict_date])
        ticks = cur.fetchall()
        
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
        
        predictions = model.predict([test_encoded, encoded_meta])
        
        ## Display ##
        
        parse = lambda num: str(round(num, 2))
        
        print("Using: " + str(test_sents))
        
        if model_type == 'regression':
            
            print("Predicting Change Coef: " +  parse(np.mean(predictions[:, 0])))
            print("Predicting Price: " +  parse(np.mean(predictions[:, 0]) * 0.0044 * ticks[0][0] + ticks[0][0]))
            
        else:
        
            print("Predicting Change Coef: " +  parse(np.mean(predictions[:, 0]) - .5))
        
        print("Actual Stock: " + parse(ticks[0][0]) + " to " + parse(ticks[-1][0]))
        print("Actual Stock Change: " + parse(ticks[-1][0] - ticks[0][0]))
            


# In[14]:

# TEST MODEL

if __name__ == "__main__":
     
    ## Load Model For Manual Testing ##
    
    import keras.metrics
    keras.metrics.correct_sign_acc = correct_sign_acc
     
    with open(os.path.join('..', 'models', 'toke.pkl'), 'rb') as toke_file:
        toke = pickle.load(toke_file)
    
    model = load_model(os.path.join('..', 'models', 'media-headlines-' + model_type + '.h5'))
      
    ## Fake Unique Test Data ##
    
    headlines = [
        "**COMPANY** gains a ton of stock after creating **PRODUCT**",
        "**COMPANY** loses a ton of stock after killing **PRODUCT**"
    ]
    
    test_sents, meta = [], []
    
    for headline in headlines:
    
        for source in all_sources:

            for weekday in range(7):
            
                test_sents.append(headline)
                meta.append([source, weekday])
    
    ## Process ##
    
    encoded_meta, test_encoded, _ = encode_sentences(meta, 
                                                     test_sents, 
                                                     tokenizer=toke, 
                                                     max_length=max_length, 
                                                     vocab_size=vocab_size)
    
    predictions = model.predict([test_encoded, encoded_meta])
    
    predictions = predictions.reshape((len(headlines), len(all_sources), 7))
    
    ## Display Predictions ##
    
    from matplotlib.colors import Normalize
    
    for i, headline in enumerate(headlines):
        
        plt.imshow(predictions[i], interpolation='none', cmap='PRGn', norm=Normalize(vmin=-2, vmax=2))
        plt.title(headline)
        plt.xlabel('Weekday')
        plt.ylabel('Source')
        plt.xticks(np.arange(7), list('MTWTFSS'))
        plt.yticks(np.arange(len(all_sources)), all_sources)
        plt.show()


# In[ ]:



