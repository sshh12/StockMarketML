
# coding: utf-8

# In[1]:

# Setup (Imports)

from sklearn.utils import shuffle

import numpy as np
import os

import matplotlib.pyplot as plt


# In[2]:

# Load CSV

def csv_as_numpy(stock):
    
    days, day_values = [], []
    
    with open(os.path.join('data', stock + '.csv'), 'r') as data:

        for line in data:

            if line.startswith('20'):

                items = line.split(",")
                
                days.append(items[0])
                day_values.append( np.array( list(map(float, items[1:])) ) )
                
    return days, np.array(day_values)


# In[3]:

# Make Data

def create_timeframed_close_regression_data(stock, window_size, window_skip=0, norm=False):
    
    data = csv_as_numpy(stock)[1][:, 3]
    
    X, Y = [], []
    
    for i in range(len(data) - window_size - 1):
        
        time_frame = np.copy(data[i: i + window_size + 1])

        if norm:
            
            mean = np.mean(time_frame[:-1])
            
            time_frame -= mean
            
            std = np.std(time_frame)
            
            time_frame /= std
            
        X.append(time_frame[:-1 - window_skip])
        Y.append(time_frame[-1])
        
    return np.array(X), np.array(Y)

def create_timeframed_alldata_classification_data(stock, window_size, norm=True, output='up/down'):
    
    data = csv_as_numpy(stock)[1][:, (0,1,2,4,5)] # OPEN HIGH LOW close ADJ_CLOSE VOLUME
    
    X, Y = [], []
    
    for i in range(1, len(data) - window_size - 1):
        
        time_frame = np.copy(data[i: i + window_size + 1])
        
        current_close = time_frame[-1, 3]
        last_close = time_frame[-2, 3]

        if norm:
            
            mean = np.mean(time_frame[:-1], axis=0)
            
            time_frame -= mean
            
            std = np.std(time_frame, axis=0)
            
            time_frame /= std
            
        X.append(time_frame[:-1])
        
        if output == 'up/down':
            
            if last_close < current_close:
            
                Y.append([1., 0.])
                
            else:
                
                Y.append([0., 1.])
                
        elif output == '+-1':

            if last_close < current_close:
            
                Y.append(+1.)
                
            else:
                
                Y.append(-1.)
        
    return np.array(X), np.array(Y)


# In[4]:

# Split Data

def split_data(X, Y, ratio=.8, mix=True):
    
    train_size = int(len(X) * ratio)
    
    trainX, testX = X[:train_size], X[train_size:]
    trainY, testY = Y[:train_size], Y[train_size:]
    
    if mix:
        
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
    
    return trainX, trainY, testX, testY


# In[5]:

# Run (Test)

if __name__ == "__main__":
    
    closing = []
    
    high, low = [], []

    for values in csv_as_numpy('AAPL')[1]:

        closing.append(values[3])
        high.append(values[1])
        low.append(values[2])

    plt.plot(closing[-50:])
    plt.plot(high[-50:])
    plt.plot(low[-50:])
    plt.show()

