
# coding: utf-8

# In[16]:

# Setup (Imports)

from LoadData import *

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, Conv1D

import numpy as np

import matplotlib.pyplot as plt


# In[17]:

# Setup (Globals/Hyperz)

window_size = 30
epochs      = 600
batch_size  = 128
emb_size    = 5


# In[18]:

# Loading and Splitting Data

def get_data(stock):
    
    AllX, AllY = create_timeframed_alldata_classification_data(stock, window_size, norm=True)
    
    trainX, trainY, testX, testY = split_data(AllX, AllY, ratio=.9)
    
    return (trainX, trainY), (testX, testY)


# In[19]:

# Setup (Create Model)

def get_model(variation='lstm'):
    
    model = Sequential()
    
    model.add(Conv1D(input_shape=(window_size, emb_size),
                     filters=16,
                     kernel_size=4,
                     padding='same', 
                     activation='relu'))

    model.add(Conv1D(filters=8,
                     kernel_size=4,
                     padding='same', 
                     activation='relu'))
 
    model.add(Flatten())
    
    model.add(Dense(64, activation='relu'))

    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        
    return model


# In[20]:

# Run (Load)

(trainX, trainY), (testX, testY) = get_data('GSPC')

print(trainX.shape, trainY.shape)


# In[21]:

# Run (Train)

model = get_model()

history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0)

plt.plot(np.log(history.history['loss']))
plt.plot(np.log(history.history['val_loss']))
plt.legend(['TrainLoss', 'TestLoss'])
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['TrainAcc', 'TestAcc'])
plt.show()

