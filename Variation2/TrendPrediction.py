
# coding: utf-8

# In[1]:

# Processing/Misc

from sklearn.utils import shuffle

import numpy as np
import os

import matplotlib.pyplot as plt


# In[2]:

# Keras

from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout, Flatten, Conv1D, BatchNormalization, Activation, GlobalMaxPooling1D, MaxPooling1D


# In[3]:

# Hyperz

epochs           = 200
batch_size       = 32
window_size      = 20
skip_window_size = 5
train_split      = .8
emb_size         = 5


# In[4]:

# Load Data

def create_timeframed_alldata_data(stocks, window_size=10, skip_window_size=2):
    """
    Timeframe Alldata
    """
    X, Y = [], []
    
    for stock in stocks:
        
        ### Load From File
        
        raw_data = []
        
        with open(os.path.join('..', 'data', stock + '.csv'), 'r') as data:

            for line in data:

                if len(line) > 6 and "Date" not in line and "null" not in line:

                    items = line.split(",")
                    items = np.array(list(map(float, items[1:])))

                    raw_data.append(np.take(items, [0, 1, 2, 4, 5])) # OPEN HIGH LOW close ADJ_CLOSE VOLUME
        
        data = np.array(raw_data)
        
        ### Process
        
        for i in range(1, len(data) - window_size - 1):

            time_frame = np.copy(data[i: i + window_size + 1])
            
            trainable_frame = time_frame[:-skip_window_size]

            target_close = time_frame[-1, 3]
            last_close = trainable_frame[-1, 3]

            time_frame -= np.mean(trainable_frame, axis=0)
            time_frame /= np.std(trainable_frame, axis=0)

            X.append(trainable_frame)

            if last_close < target_close:

                Y.append([1., time_frame[-1, 3]])

            else:

                Y.append([0., time_frame[-1, 3]])
        
    return np.array(X), np.array(Y)


# In[5]:

# Split

def split_data(X, Y, ratio=.8, mix=True):
    """
    Splits X/Y to Train/Test
    """
    train_size = int(len(X) * ratio)
    
    trainX, testX = X[:train_size], X[train_size:]
    trainY, testY = Y[:train_size], Y[train_size:]
    
    if mix:
        
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
    
    return trainX, trainY, testX, testY

def get_data(stocks):
    
    X, Y = create_timeframed_alldata_data(stocks, window_size=window_size, skip_window_size=skip_window_size)
    
    return split_data(X, Y, ratio=train_split)


# In[20]:

# Model

def get_model():
    
    model = Sequential()

    model.add(LSTM(200, input_shape=(window_size - skip_window_size + 1, emb_size)))
    
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(80))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(2))
    
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        
    return model


# In[ ]:

# Load Data

if __name__ == "__main__":
    
    trainX, trainY, testX, testY = get_data(['AAPL'])


# In[ ]:

# Train

if __name__ == "__main__":

    model = get_model()

    reduce_LR = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=1e-6, verbose=0)
    e_stopping = EarlyStopping(patience=50)
    checkpoint = ModelCheckpoint(os.path.join('..', 'models', 'basic-classification.h5'), 
                                 monitor='val_acc', 
                                 verbose=0, 
                                 save_best_only=True)

    history = model.fit(trainX, trainY, epochs=epochs, 
                                        batch_size=batch_size, 
                                        validation_data=(testX, testY), 
                                        verbose=0, 
                                        callbacks=[])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['TrainLoss', 'TestLoss'])
    plt.show()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['TrainAcc', 'TestAcc'])
    plt.show()

