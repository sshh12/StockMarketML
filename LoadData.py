
# coding: utf-8

# In[1]:

# Setup (Imports)

import numpy as np
import os

import matplotlib.pyplot as plt


# In[2]:

# Functions

def csv_as_numpy(stock):
    
    days, day_values = [], []
    
    with open(os.path.join('data', stock + '.csv'), 'r') as data:

        for line in data:

            if line.startswith('20'):

                items = line.split(",")
                
                days.append(items[0])
                day_values.append( np.array( list(map(float, items[1:])) ) )
                
    return days, np.array(day_values)

def create_chunks(data, window_size):
    
    X, Y = [], []
    
    for i in range(len(data) - window_size - 1):
        
        X.append(data[i: i + window_size])
        Y.append(data[i + window_size])
        
    return np.array(X), np.array(Y)


# In[3]:

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

