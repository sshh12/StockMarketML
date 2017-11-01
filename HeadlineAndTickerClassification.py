
# coding: utf-8

# In[1]:

# Setup (Imports)

from LoadData import *

from keras.layers import concatenate, Concatenate
from keras.models import load_model

import numpy as np
import os


# In[2]:

# Setup (Globals/Hyperz)

window_size  = 6
epochs       = 750
batch_size   = 128
emb_size     = 100


# In[3]:

# Make Model

def get_model():
    
    ticker_model = load_model(os.path.join('models', 'basic-classification.h5'))
    headline_model = load_model(os.path.join('models', 'headline-classification.h5'))
    
    ticker_model.pop()
    headline_model.pop()
    
    combined = concatenate([ticker_model.outputs, headline_model.outputs])
    
    combined.add(Dense(16))
    combined.add(BatchNormalization())
    combined.add(Activation('relu'))
    combined.add(Dropout(0.25))

    combined.add(Dense(2, activation='softmax'))
    
    print(combined.summary())
    
    # TODO: How to combine models

get_model()

