
# coding: utf-8

# In[9]:

# Setup (Imports)

from LoadData import *

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

import numpy as np

import matplotlib.pyplot as plt


# In[10]:

# Setup (Globals/Hyperz)

window_size = 30
epochs      = 600
batch_size  = 64


# In[11]:

# Loading and Splitting Data

def get_data(stock, ratio=.80, variation='lstm'):
    
    data = csv_as_numpy(stock)[1][:, 3] # 3 = Closing Price
    
    train_size = int(len(data) * ratio)
    
    if variation == 'lstm':
    
        trainX, trainY = create_chunks(data[: train_size], window_size, norm=True)

        testX, testY = create_chunks(data[train_size:], window_size, norm=True)

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
    if variation == 'mlp':
    
        trainX, trainY = create_chunks(data[: train_size], window_size, norm=True)

        testX, testY = create_chunks(data[train_size:], window_size, norm=True)
    
    return (trainX, trainY), (testX, testY)


# In[12]:

# Setup (Create Model)

def get_model(variation='lstm'):
    
    if variation == 'lstm':

        model = Sequential()

        model.add(LSTM(20, input_shape=(1, window_size), return_sequences=True))

        model.add(LSTM(20, return_sequences=False))

        model.add(Dense(10, activation='relu'))

        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')
        
    elif variation == 'mlp':
        
        model = Sequential()

        model.add(Dense(20, activation='relu'))
        model.add(Dropout(.5))
        
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(.5))
        
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(.5))
        
        model.add(Dense(20, activation='relu'))

        model.compile(loss='mse', optimizer='adam')
    
    return model


# In[13]:

# Run (Load)

(trainX, trainY), (testX, testY) = get_data('AAPL', variation='mlp')

print(trainX.shape, trainY.shape)


# In[14]:

# Run (Train)

model = get_model(variation='mlp')

history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['TrainLoss', 'TestLoss'])
plt.show()


# In[ ]:

# Test

data = csv_as_numpy('AAPL')[1][:, 3]

data = data[:]

prices_actual = []
prices_predicted = []

for i in range(len(data) - window_size - 1):
        
    X = data[i: i + window_size]
    Y = data[i + window_size]
    
    prices_actual.append(Y)
    
    X = np.array([X])
    
    mean, std = np.mean(X), np.std(X) 
    
    X = X - mean / std
    
    prediction = model.predict(np.reshape(X, (X.shape[0], 1, X.shape[1])))
    
    prediction = prediction * std + mean
    
    prices_predicted.append(np.squeeze(prediction))

plt.plot(prices_actual)
plt.plot(prices_predicted)
plt.show()

