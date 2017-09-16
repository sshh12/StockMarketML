
# coding: utf-8

# In[ ]:

# Setup (Imports)

from LoadData import *

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

import numpy as np


# In[ ]:

# Setup (Globals/Hyperz)

window_size = 20
epochs      = 100
batch_size  = 1


# In[ ]:

# Loading and Splitting Data

def get_data(stock, ratio=.67):
    
    data = csv_as_numpy(stock)[1][:, 3]
    
    # Scale
    
    scaler = StandardScaler(with_mean=False)
    
    data = np.squeeze(scaler.fit_transform(np.array([data])))
    
    # Split
    
    train_size = int(len(data) * ratio)
    
    trainX, trainY = create_chunks(data[: train_size], window_size)
    
    testX, testY = create_chunks(data[train_size:], window_size)
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    return (trainX, trainY), (testX, testY), scaler


# In[ ]:

# Setup (Create Model)

def get_model():

    model = Sequential()
    
    model.add(LSTM(4, input_shape=(1, window_size)))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model


# In[ ]:

# Run (Load)

(trainX, trainY), (testX, testY), scaler = get_data('AAPL')

print(trainX.shape, trainY.shape)


# In[ ]:

# Run (Train)

model = get_model()

model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

